import pytest

from dudamel import App, Orchestrator
from dudamel.exceptions import RegistryError
from dudamel.registry import Registry


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
