import pytest

from dudamel import App
from dudamel.exceptions import RegistryError, RuntimeNotBoundError


def make_app() -> App:
    return App("workouts", description="Log workouts")


def test_bare_decorator_registers_tool():
    app = make_app()

    @app.tool
    async def log_workout(exercise: str) -> str:
        """Record one exercise."""
        return exercise

    t = app.tools["log_workout"]
    assert t.app_name == "workouts"
    assert t.description == "Record one exercise."
    assert t.read_only is False and t.confirm is False and t.timeout == 30.0


def test_decorator_with_flags():
    app = make_app()

    @app.tool(read_only=True, confirm=True, timeout=5.0)
    async def get_stats() -> str:
        """Read stats."""
        return "ok"

    t = app.tools["get_stats"]
    assert t.read_only is True and t.confirm is True and t.timeout == 5.0


def test_docstring_required():
    app = make_app()
    with pytest.raises(RegistryError, match="docstring"):

        @app.tool
        async def nameless(x: int) -> int:
            return x


def test_duplicate_tool_rejected():
    app = make_app()

    @app.tool
    async def dup() -> str:
        """One."""
        return "1"

    with pytest.raises(RegistryError, match="already registered"):
        app._register_tool(dup, read_only=False, confirm=False, timeout=30.0)


def test_sync_tool_rejected():
    app = make_app()
    with pytest.raises(RegistryError, match="must be async"):

        @app.tool
        def sync_fn(x: int) -> int:
            """Doc."""
            return x


def test_tool_var_args_wrapped_as_registry_error():
    """ToolSchema raises TypeError for *args/**kwargs (tests/test_schema.py
    tests that directly); the decorator path must fold it into RegistryError
    so registration failures are one taxonomy end to end."""
    app = make_app()
    with pytest.raises(RegistryError, match=r"\*args/\*\*kwargs not supported"):

        @app.tool
        async def var_args(*args: int) -> int:
            """Doc."""
            return sum(args)


def test_bad_tool_name_rejected():
    app = make_app()

    async def bad(x: int) -> int:
        """Doc."""
        return x

    bad.__name__ = "has.dots"  # dots are illegal in provider tool-name regex
    with pytest.raises(RegistryError, match="tool name"):
        app._register_tool(bad, read_only=False, confirm=False, timeout=30.0)


async def test_llm_and_notify_unbound_raise():
    app = make_app()
    with pytest.raises(RuntimeNotBoundError):
        await app.llm("hi")
    with pytest.raises(RuntimeNotBoundError):
        await app.notify("hi")


def test_bad_app_names_rejected():
    with pytest.raises(RegistryError, match="app name"):
        App("has_underscore", description="d")
    with pytest.raises(RegistryError, match="app name"):
        App("Has-Hyphen", description="d")
    with pytest.raises(RegistryError, match="app name"):
        App("1starts", description="d")


async def test_to_thread_runs_sync_fn():
    app = make_app()
    result = await app.to_thread(lambda: 41 + 1)
    assert result == 42


def test_tool_defaults_to_not_external():
    app = App("notes", description="d")

    @app.tool(read_only=True)
    async def count_notes() -> str:
        """Count notes."""
        return "0"

    assert app.tools["count_notes"].external is False
    assert app.tools["count_notes"].untrusted is False


def test_external_flag_is_recorded_and_makes_the_tool_untrusted():
    app = App("feeds", description="d")

    @app.tool(read_only=True, external=True)
    async def read_feed() -> str:
        """Read a syndicated feed."""
        return "content"

    tool = app.tools["read_feed"]
    assert tool.external is True
    assert tool.untrusted is True
    # external is orthogonal to read_only: a fetch stays read-only, and it is
    # the caller's *later* mutations that get gated, not this call.
    assert tool.read_only is True
    assert tool.confirm is False


def test_mcp_origin_is_untrusted_even_without_the_external_flag():
    app = App("notes", description="d")

    @app.tool
    async def save_note(text: str) -> str:
        """Save a note."""
        return "saved"

    tool = app.tools["save_note"]
    assert tool.untrusted is False
    # Assigned AFTER construction, which is how mcp_mount force-gates tools in
    # place and how the confirm tests build an mcp-provided tool. A value
    # computed at construction time would be stale here.
    tool.origin = "mcp"
    assert tool.untrusted is True
    assert tool.external is False  # the declared flag is untouched


def test_tool_action_label_is_stored_and_stripped() -> None:
    app = App("tasks", description="d")

    @app.tool(action="  Done  ")
    async def complete(id: int) -> str:
        """Complete a task."""
        return "ok"

    assert app.tools["complete"].action == "Done"


def test_tool_without_action_has_none() -> None:
    app = App("tasks", description="d")

    @app.tool
    async def plain(id: int) -> str:
        """Plain."""
        return "ok"

    assert app.tools["plain"].action is None


def test_blank_action_label_is_rejected() -> None:
    app = App("tasks", description="d")
    with pytest.raises(RegistryError, match="must not be empty"):

        @app.tool(action="   ")
        async def complete(id: int) -> str:
            """Complete a task."""
            return "ok"


def test_overlong_action_label_is_rejected() -> None:
    app = App("tasks", description="d")
    with pytest.raises(RegistryError, match="at most 32 characters"):

        @app.tool(action="x" * 33)
        async def complete(id: int) -> str:
            """Complete a task."""
            return "ok"


def test_action_label_of_exactly_the_limit_is_accepted() -> None:
    """32 characters is the longest label that fits, not the first that does
    not -- pinned from the accepting side so the comparison cannot quietly
    tighten to `>=`."""
    app = App("tasks", description="d")
    label = "x" * 32

    @app.tool(action=label)
    async def complete(id: int) -> str:
        """Complete a task."""
        return "ok"

    assert app.tools["complete"].action == label


def test_action_label_is_stripped_before_the_length_check() -> None:
    """Padding is not length: a label that only exceeds the limit because of
    surrounding whitespace is accepted, and stored stripped."""
    app = App("tasks", description="d")

    @app.tool(action="  " + "x" * 32)
    async def complete(id: int) -> str:
        """Complete a task."""
        return "ok"

    assert app.tools["complete"].action == "x" * 32


def test_bidi_and_control_characters_are_stripped_from_a_registered_label() -> None:
    """A label is drawn on a button, put into a confirm dialog and read out by
    a screen reader; a direction override makes all three read as something
    other than the tool they run. Neutralized where the label is accepted, so
    an HTML surface and a plain-text one cannot disagree about what a button
    says."""
    app = App("tasks", description="d")

    @app.tool(action="Nuke\u202e evihcrA\u2069\x07")
    async def complete(id: int) -> str:
        """Complete a task."""
        return "ok"

    assert app.tools["complete"].action == "Nuke evihcrA"


def test_a_label_is_sanitized_before_its_length_is_measured() -> None:
    """Ordering, not cosmetics. Measuring first would approve a label at the
    cap and then store a shorter one, and would store a label spelled entirely
    out of overrides as the empty string the non-empty rule exists to refuse."""
    app = App("tasks", description="d")

    @app.tool(action="x" * 32 + "\u202e\u202a")
    async def complete(id: int) -> str:
        """Complete a task."""
        return "ok"

    assert app.tools["complete"].action == "x" * 32

    other = App("chores", description="d")
    with pytest.raises(RegistryError, match="must not be empty"):

        @other.tool(action="\u202e\u2066")
        async def wipe(id: int) -> str:
            """Delete a task."""
            return "ok"


def test_widget_actions_default_to_empty_tuple() -> None:
    app = App("tasks", description="d")

    @app.widget(title="T", renderer="markdown")
    async def card() -> str:
        return "hi"

    assert app.widgets["card"].actions == ()


def test_widget_qualified_id_is_app_dot_widget() -> None:
    app = App("tasks", description="d")

    @app.widget(title="T", renderer="markdown")
    async def today() -> str:
        return "hi"

    assert app.widgets["today"].qualified_id == "tasks.today"
