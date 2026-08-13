import pytest

from dudamel import App, Orchestrator
from dudamel.contract.schema import ToolSchema
from dudamel.contract.types import Tool
from dudamel.exceptions import RegistryError
from dudamel.registry import Registry


async def _noop() -> str:
    """Noop."""
    return "ok"


def app_with_tool(app_name: str, tool_name: str) -> App:
    app = App(app_name, description="d")

    async def fn() -> str:
        """Doc."""
        return "x"

    fn.__name__ = tool_name
    app._register_tool(fn, read_only=False, confirm=False, timeout=30.0)
    return app


def test_registry_collects_everything():
    a = app_with_tool("alpha", "do_a")
    b = app_with_tool("beta", "do_b")
    r = Registry([a, b])
    assert set(r.tools) == {"do_a", "do_b"}
    assert set(r.apps) == {"alpha", "beta"}
    assert set(r.metadatas) == {"alpha", "beta"}


def test_cross_app_tool_collision_rejected():
    with pytest.raises(RegistryError, match="do_x"):
        Registry([app_with_tool("alpha", "do_x"), app_with_tool("beta", "do_x")])


def test_duplicate_app_name_rejected():
    with pytest.raises(RegistryError, match="alpha"):
        Registry([App("alpha", description="1"), App("alpha", description="2")])


def test_reserved_app_name_rejected():
    with pytest.raises(RegistryError, match="reserved"):
        Registry([App("core", description="d")])


@pytest.mark.parametrize("name", ["job", "alembic", "pending"])
def test_core_namespace_collision_rejected(name):
    """App("job") would prefix its tables "job_" -> collides with the core
    table "job_runs"; App("pending") collides with "pending_confirmations";
    App("alembic") collides with the alembic version-table namespace itself.
    All three must be rejected at Registry construction, before any table
    ever gets a chance to be reflected/diffed by migrate.py."""
    with pytest.raises(RegistryError, match="core namespace"):
        Registry([App(name, description="d")])


def test_orchestrator_is_pure():
    a = app_with_tool("alpha", "do_a")
    orc = Orchestrator(apps=[a], mcp=["uvx mcp-server-fetch"])
    assert orc.registry.tools["do_a"].app_name == "alpha"
    assert orc.mcp == ["uvx mcp-server-fetch"]  # stored, not launched
    # purity: no scheduler, no subprocesses, no event loop, no db binding
    assert a._database is None


def _labelled_app() -> App:
    app = App("tasks", description="d")

    @app.tool(action="Refresh")
    async def refresh() -> str:
        """Refresh."""
        return "ok"

    return app


def test_card_action_naming_a_labelled_zero_arg_tool_is_accepted() -> None:
    app = _labelled_app()

    @app.widget(title="T", renderer="markdown", actions=["refresh"])
    async def card() -> str:
        return "hi"

    registry = Registry([app])
    assert registry.widgets[0].actions == ("refresh",)


def test_card_action_naming_an_unknown_tool_is_rejected() -> None:
    app = App("tasks", description="d")

    @app.widget(title="T", renderer="markdown", actions=["nope"])
    async def card() -> str:
        return "hi"

    with pytest.raises(RegistryError, match="not an action-labelled tool"):
        Registry([app])


def test_card_action_naming_an_unlabelled_tool_is_rejected() -> None:
    app = App("tasks", description="d")

    @app.tool
    async def plain() -> str:
        """Plain."""
        return "ok"

    @app.widget(title="T", renderer="markdown", actions=["plain"])
    async def card() -> str:
        return "hi"

    with pytest.raises(RegistryError, match="has no action= label"):
        Registry([app])


def test_card_action_with_a_required_parameter_is_rejected() -> None:
    app = App("tasks", description="d")

    @app.tool(action="Do")
    async def needs_arg(id: int) -> str:
        """Needs an arg."""
        return "ok"

    @app.widget(title="T", renderer="markdown", actions=["needs_arg"])
    async def card() -> str:
        return "hi"

    with pytest.raises(RegistryError, match="required parameter"):
        Registry([app])


def test_card_action_naming_another_apps_tool_is_rejected() -> None:
    other = _labelled_app()
    mine = App("notes", description="d")

    @mine.widget(title="T", renderer="markdown", actions=["refresh"])
    async def card() -> str:
        return "hi"

    with pytest.raises(RegistryError, match="not an action-labelled tool"):
        Registry([other, mine])


def test_mcp_tool_may_not_carry_an_action_label() -> None:
    registry = Registry([])
    tool = Tool(
        name="srv__thing",
        app_name="srv",
        description="d",
        fn=_noop,
        schema=ToolSchema(_noop),
        read_only=True,
        confirm=False,
        timeout=1.0,
        origin="mcp",
        action="Click me",
    )
    with pytest.raises(RegistryError, match="action label"):
        registry.add_mcp_tools([tool])
