"""Fixture-behavior tests for `mcp_flaky_server.py` (experimental MCP mount).

These prove the flaky fixture itself does what a reconnect implementation
needs from it -- dies on demand, records side effects that survive a
restart, and can advertise a different tool surface across restarts. They
spawn a real subprocess per `MCPMount`, same as `test_mcp_mount.py`.

Every test wraps its `mount.close()` in `try/finally`: a failed assertion
before `close()` would otherwise leak the spawned subprocess and hang
pytest in teardown (see `tests/fixtures/mcp_flaky_server.py`'s sibling
`mcp_echo_server.py` tests for the same hazard).
"""

from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path

import pytest

from dudamel.mcp_mount import MCPMount, MCPServerConfig

FIXTURE = Path(__file__).parent / "fixtures" / "mcp_flaky_server.py"


async def _wait_for_started_marker(state: Path, value: str, *, timeout: float = 5.0) -> None:
    """Poll `state` for slow_mutate's `started:<value>` marker line -- proof
    the handler is actually running in its thread, not just that the call
    was fired. A fixed sleep here would be a race: real CI schedulers can
    add multi-second latency between firing an async call and its handler
    thread actually starting, and there's no other way to observe that from
    outside the subprocess. Bounded: raises with a clear message instead of
    hanging if the marker never shows up.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    marker = f"started:{value}"
    while loop.time() < deadline:
        if state.exists() and marker in state.read_text().splitlines():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out after {timeout}s waiting for {marker!r} to appear in {state}")


def flaky_cmd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    state: Path | None = None,
    drift: bool = False,
    drop: tuple[str, ...] = (),
    slow_seconds: float = 0.05,
    name: str | None = None,
) -> MCPServerConfig:
    """Build an `MCPServerConfig` that launches the flaky fixture with the
    given behavior, configured entirely through environment variables.

    The fixture reads its config from its OWN environment at import time,
    and `MCPServerConfig.env` only forwards variables that already exist in
    THIS (parent) process -- so the variables are set here via `monkeypatch`
    (auto-restored at test teardown) and then explicitly listed in `env=` so
    the mount actually passes them down to the subprocess.
    """
    argv = [sys.executable, str(FIXTURE)]
    if name is not None:
        argv.append(name)
    state_path = state if state is not None else tmp_path / "flaky-state"
    monkeypatch.setenv("MCP_FLAKY_STATE", str(state_path))
    monkeypatch.setenv("MCP_FLAKY_SLOW_SECONDS", str(slow_seconds))
    if drift:
        monkeypatch.setenv("MCP_FLAKY_ANNOTATIONS", "drift")
    else:
        monkeypatch.delenv("MCP_FLAKY_ANNOTATIONS", raising=False)
    if drop:
        monkeypatch.setenv("MCP_FLAKY_DROP", ",".join(drop))
    else:
        monkeypatch.delenv("MCP_FLAKY_DROP", raising=False)
    return MCPServerConfig(
        command=shlex.join(argv),
        env=(
            "MCP_FLAKY_STATE",
            "MCP_FLAKY_SLOW_SECONDS",
            "MCP_FLAKY_ANNOTATIONS",
            "MCP_FLAKY_DROP",
        ),
    )


async def test_flaky_fixture_dies_on_demand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch)])
    try:
        tools = await mount.mount()
        die = next(t for t in tools if t.name.endswith("__die"))
        with pytest.raises(BaseException):  # noqa: B017 -- type varies: MCPError, anyio errors, or CancelledError
            await die.fn()
    finally:
        await mount.close()


async def test_flaky_fixture_records_side_effects_across_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    for _ in range(2):
        mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, state=state)])
        try:
            tools = await mount.mount()
            mutate = next(t for t in tools if t.name.endswith("__slow_mutate"))
            assert await mutate.fn(value="x") == "mutated:x"
        finally:
            await mount.close()
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, state=state)])
    try:
        tools = await mount.mount()
        count = next(t for t in tools if t.name.endswith("__count"))
        assert await count.fn() == "2"
    finally:
        await mount.close()


async def test_flaky_fixture_records_side_effect_even_if_killed_before_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The side effect is written (and synced) before the sleep that a test
    can race a `die()` against -- so state is on disk even if the reply
    never arrives. This is the property the next task's mid-call-kill test
    depends on. Killing is gated on the `started:` marker actually
    appearing on disk, not a fixed sleep -- see `_wait_for_started_marker`."""
    state = tmp_path / "state"
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, state=state, slow_seconds=10.0)])
    try:
        tools = await mount.mount()
        mutate = next(t for t in tools if t.name.endswith("__slow_mutate"))
        die = next(t for t in tools if t.name.endswith("__die"))
        # Fire the mutation but don't wait for its (slow) reply; instead
        # kill the process once we have proof (the started marker) that the
        # side effect has landed, even though the mutate call never returns.
        mutate_call = asyncio.ensure_future(mutate.fn(value="y"))
        await _wait_for_started_marker(state, "y")
        with pytest.raises(BaseException):  # noqa: B017 -- type varies: MCPError, anyio errors, or CancelledError
            await die.fn()
        # The dying process breaks the still-pending mutate call's transport
        # too -- it never gets a reply, exactly the case a naive retry would
        # double-execute. The side effect is still on disk (asserted below),
        # proving it landed before the process died.
        with pytest.raises(BaseException):  # noqa: B017 -- type varies: MCPError, anyio errors, or CancelledError
            await mutate_call
    finally:
        await mount.close()
    assert state.read_text().splitlines() == ["started:y", "y"]

    # A restarted server must report exactly one COMPLETED mutation, not
    # two -- if count() mistakenly counted the "started:" marker as well as
    # the real line, it would silently report double execution that never
    # happened, corrupting the very proof the next task depends on.
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, state=state)])
    try:
        tools = await mount.mount()
        count = next(t for t in tools if t.name.endswith("__count"))
        assert await count.fn() == "1"
    finally:
        await mount.close()


async def test_flaky_fixture_can_drift_its_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, drift=True)])
    try:
        tools = await mount.mount()
        echo = next(t for t in tools if t.name.endswith("__echo"))
        assert echo.read_only is False
    finally:
        await mount.close()


async def test_flaky_fixture_undrifted_echo_stays_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch)])
    try:
        tools = await mount.mount()
        echo = next(t for t in tools if t.name.endswith("__echo"))
        assert echo.read_only is True
    finally:
        await mount.close()


async def test_flaky_fixture_can_drop_a_tool_entirely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, drop=("echo",))])
    try:
        tools = await mount.mount()
        names = {t.name for t in tools}
        assert not any(n.endswith("__echo") for n in names)
        assert any(n.endswith("__die") for n in names)
        assert any(n.endswith("__slow_mutate") for n in names)
        assert any(n.endswith("__count") for n in names)
    finally:
        await mount.close()


async def test_flaky_fixture_advertises_all_four_tools_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch)])
    try:
        tools = await mount.mount()
        suffixes = {t.name.rsplit("__", 1)[-1] for t in tools}
        assert suffixes == {"die", "slow_mutate", "count", "echo"}
    finally:
        await mount.close()
