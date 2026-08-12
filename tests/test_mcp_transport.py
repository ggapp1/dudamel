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

from dudamel import Orchestrator, Runtime, mcp_mount
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


def test_http_label_strips_userinfo_and_query() -> None:
    """URLs carry secrets in userinfo and query strings; scheme/host/path are
    enough to identify a server in a log line."""
    cfg = MCPServerConfig(url="https://user:pw@host/mcp?token=sk-secret-123")
    assert "sk-secret-123" not in cfg.label
    assert "pw" not in cfg.label
    assert "host" in cfg.label


def test_mcp_server_rejects_neither_command_nor_url() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        MCPServerConfig()


def test_mcp_server_rejects_both_command_and_url() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        MCPServerConfig(command="x", url="http://localhost/mcp")


async def test_unreachable_url_yields_no_tools_and_never_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Port 9 is the discard port -- reliably refuses. The caplog assertion
    pins this to the real degrade-and-skip code path in mount()'s except
    block, not just "no tools came back for some other reason"."""
    mount = MCPMount([MCPServerConfig(url="http://127.0.0.1:9/mcp")])
    assert await mount.mount() == []
    await mount.close()
    assert any(
        "failed to mount" in r.message and "127.0.0.1:9" in r.message for r in caplog.records
    )


async def test_headers_reach_the_http_client_and_never_the_logs(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The auth path, end to end at the seam that matters: `headers` exists so
    a bare URL can reach an authenticated server, and the only thing that makes
    that work is `create_mcp_http_client(headers=...)` receiving the dict
    unchanged. The same secret must never come back out in a log line -- the
    URL is unreachable here, so the mount-failure warning fires and pins the
    redaction for the HTTP transport specifically.
    """
    headers = {"Authorization": "Bearer sk-secret-789", "X-Tenant": "acme"}
    captured: list[dict[str, str] | None] = []
    real_client = mcp_mount.create_mcp_http_client

    def capturing_client(*args: object, **kwargs: object) -> object:
        captured.append(kwargs.get("headers"))  # type: ignore[arg-type]
        return real_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(mcp_mount, "create_mcp_http_client", capturing_client)
    # Port 9 (discard) refuses, so the mount degrades -- after the transport
    # has already been built with these headers.
    mount = MCPMount([MCPServerConfig(url="http://127.0.0.1:9/mcp", headers=headers)])
    tools = await mount.mount()
    await mount.close()
    assert tools == []
    assert captured == [headers]
    # Verbatim: the same pairs, not a re-cased or filtered copy.
    assert captured[0] is not None and dict(captured[0]) == headers
    assert not any("sk-secret-789" in r.message for r in caplog.records)
    assert not any("Authorization" in r.message for r in caplog.records)
    assert any(
        "failed to mount" in r.message and "127.0.0.1:9" in r.message for r in caplog.records
    )
    labelled = MCPServerConfig(url="http://127.0.0.1:9/mcp", headers=headers)
    assert "sk-secret-789" not in labelled.label


async def test_empty_string_entry_degrades_instead_of_crashing_construction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A "" entry (e.g. os.environ.get("SOME_MCP_CMD", "")) must degrade to
    zero tools like any other bad config, not raise out of MCPMount() /
    mount() -- MCPServerConfig("") fails "exactly one of command/url", and
    that construction now happens inside mount()'s per-server try, not in
    __init__ where it would crash Runtime.start() before any server ran."""
    mount = MCPMount([""])
    assert await mount.mount() == []
    await mount.close()
    assert any("failed to mount" in r.message for r in caplog.records)


async def test_dataclass_entry_does_not_inherit_global_env_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCPServerConfig(command=...) with no `env` of its own must NOT pick up
    MCPMount(env_passthrough=...) -- that global list applies only to plain
    string entries; a dataclass entry's own (empty, here) `env` wins."""
    monkeypatch.setenv("DUDAMEL_TEST_VAR", "leaked-value")
    mount = MCPMount(
        [MCPServerConfig(command=FIXTURE_COMMAND)], env_passthrough=("DUDAMEL_TEST_VAR",)
    )
    try:
        tools = await mount.mount()
        read_env = next(t for t in tools if t.name.endswith("__read_env"))
        assert await read_env.fn(name="DUDAMEL_TEST_VAR") == ""
    finally:
        await mount.close()


async def test_runtime_start_succeeds_with_unreachable_http_server(tmp_path: Path) -> None:
    orc = Orchestrator(apps=[], mcp=[MCPServerConfig(url="http://127.0.0.1:9/mcp")])
    rt = Runtime(
        orc, make_settings(tmp_path), providers={"standard": FakeProvider([fake_text("hi")])}
    )
    await rt.start()  # must NOT raise -- broken mcp server never breaks core
    try:
        assert orc.registry.tools == {}
        reply = await rt.chat("t:1", "hi", user_id="u1")
        assert reply.text == "hi"
    finally:
        await rt.stop()


async def test_string_entries_still_mount_as_stdio_commands() -> None:
    mount = MCPMount([fixture_cmd("alpha")])
    try:
        tools = await mount.mount()
        assert {t.name for t in tools} >= {"alpha__echo"}
    finally:
        await mount.close()


async def test_per_server_env_reaches_the_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """MCPServerConfig(env=...) replaces the global env_passthrough list."""
    monkeypatch.setenv("DUDAMEL_TEST_VAR", "per-server-value")
    mount = MCPMount([MCPServerConfig(command=FIXTURE_COMMAND, env=("DUDAMEL_TEST_VAR",))])
    try:
        tools = await mount.mount()
        read_env = next(t for t in tools if t.name.endswith("__read_env"))
        assert await read_env.fn(name="DUDAMEL_TEST_VAR") == "per-server-value"
    finally:
        await mount.close()
