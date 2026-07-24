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
