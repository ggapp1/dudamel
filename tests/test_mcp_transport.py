"""HTTP-transport mounting for MCP servers.

`MCPServerConfig` lets an `Orchestrator(mcp=[...])` entry carry more than a
stdio command string: an HTTP URL with headers, or per-server environment
passthrough. A plain string still means "stdio command", unchanged.

The unreachable-URL case is the important one: mcp 2.0.0's
`streamable_http_client` defers the real connection failure into a native
`ExceptionGroup` wrapping `httpx2.ConnectError` -- `Exception`-derived, so a
plain `except Exception` already catches it -- but a `ClientSession` whose
background task group has already failed emits a bare `CancelledError` (a
`BaseException`) on its *next* operation. `mount()` does operate on sessions
(`initialize()`, `list_tools()`), so its handler is widened to `except
BaseException` for that reason. Either way, an unreachable server must
degrade to zero tools, never a raised exception out of `mount()`.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

from dudamel import Orchestrator, Runtime
from dudamel.config import McpConfig, RouterConfig, Settings, TierConfig
from dudamel.llm.testing import FakeProvider, fake_text
from dudamel.mcp_mount import MCPMount, MCPServerConfig

FIXTURE = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
FIXTURE_COMMAND = shlex.join([sys.executable, str(FIXTURE)])


def fixture_cmd(name: str) -> str:
    """A fixture command that self-reports serverInfo.name = `name` instead
    of the default "fixture"."""
    return shlex.join([sys.executable, str(FIXTURE), name])


def make_settings(
    tmp_path: Path,
    *,
    router: RouterConfig | None = None,
    mcp: McpConfig | None = None,
    **tiers: TierConfig,
) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/mcp.db",
        data_dir=tmp_path,
        llm_tiers=tiers or {"standard": TierConfig(provider="fake", model="f")},
        router=router or RouterConfig(),
        mcp=mcp or McpConfig(),
    )


def test_mcp_server_rejects_neither_command_nor_url() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        MCPServerConfig()


def test_mcp_server_rejects_both_command_and_url() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        MCPServerConfig(command="x", url="http://localhost/mcp")


async def test_unreachable_url_yields_no_tools_and_never_raises() -> None:
    """Port 9 is the discard port -- reliably refuses."""
    mount = MCPMount([MCPServerConfig(url="http://127.0.0.1:9/mcp")])
    assert await mount.mount() == []
    await mount.close()


async def test_runtime_start_succeeds_with_unreachable_http_server(tmp_path: Path) -> None:
    orc = Orchestrator(apps=[], mcp=[MCPServerConfig(url="http://127.0.0.1:9/mcp")])
    rt = Runtime(
        orc, make_settings(tmp_path), providers={"standard": FakeProvider([fake_text("hi")])}
    )
    await rt.start()  # must NOT raise -- broken mcp server never breaks core
    assert orc.registry.tools == {}
    reply = await rt.chat("t:1", "hi", user_id="u1")
    assert reply.text == "hi"
    await rt.stop()


async def test_string_entries_still_mount_as_stdio_commands() -> None:
    mount = MCPMount([fixture_cmd("alpha")])
    tools = await mount.mount()
    assert {t.name for t in tools} >= {"alpha__echo"}
    await mount.close()


async def test_per_server_env_reaches_the_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """MCPServerConfig(env=...) replaces the global env_passthrough list."""
    monkeypatch.setenv("DUDAMEL_TEST_VAR", "per-server-value")
    mount = MCPMount([MCPServerConfig(command=FIXTURE_COMMAND, env=("DUDAMEL_TEST_VAR",))])
    tools = await mount.mount()
    read_env = next(t for t in tools if t.name.endswith("__read_env"))
    assert await read_env.fn(name="DUDAMEL_TEST_VAR") == "per-server-value"
    await mount.close()
