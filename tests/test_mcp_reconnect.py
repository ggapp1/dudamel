"""Reconnect behavior for the experimental MCP mount, plus the flaky
fixture's own guarantees that those tests are built on.

The first half proves `mcp_flaky_server.py` does what reconnect testing
needs from it -- dies on demand, records side effects that survive a
restart, and can advertise a different tool surface across restarts. The
second half drives the real reconnect path: a server is mounted, its
process is `SIGKILL`ed out from under a live session, and the tool call
that trips over the corpse is expected to rebuild the connection.

Killing is done by pid, found with `pgrep` against a per-test server name,
rather than through the fixture's own `die()` tool. `die()` is itself a
tool call, so calling it would trigger the very reconnect the test is
trying to observe; an external kill leaves the mount believing it is still
healthy, which is the real-world shape of a server that crashes.

They spawn a real subprocess per `MCPMount`, same as `test_mcp_mount.py`.

Every test wraps its `mount.close()` in `try/finally`: a failed assertion
before `close()` would otherwise leak the spawned subprocess and hang
pytest in teardown (see `tests/fixtures/mcp_flaky_server.py`'s sibling
`mcp_echo_server.py` tests for the same hazard).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import mcp.types as types
import pytest
from mcp.shared.exceptions import MCPError

from dudamel import mcp_mount
from dudamel.contract.types import Tool
from dudamel.exceptions import UnknownToolOutcome
from dudamel.mcp_mount import (
    ROUTER_TIMEOUT_MARGIN_SECONDS,
    MCPMount,
    MCPServerConfig,
    _is_connection_death,
    _MountedServer,
)

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
    script: Path | None = None,
) -> MCPServerConfig:
    """Build an `MCPServerConfig` that launches the flaky fixture with the
    given behavior, configured entirely through environment variables.

    The fixture reads its config from its OWN environment at import time,
    and `MCPServerConfig.env` only forwards variables that already exist in
    THIS (parent) process -- so the variables are set here via `monkeypatch`
    (auto-restored at test teardown) and then explicitly listed in `env=` so
    the mount actually passes them down to the subprocess.
    """
    argv = [sys.executable, str(script or FIXTURE)]
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
    never arrives. This is the property the mid-call-kill reconnect tests
    below depend on. Killing is gated on the `started:` marker actually
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
    # happened, corrupting the very proof the reconnect tests rest on.
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


async def test_flaky_fixture_advertises_all_five_tools_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch)])
    try:
        tools = await mount.mount()
        suffixes = {t.name.rsplit("__", 1)[-1] for t in tools}
        assert suffixes == {"die", "slow_mutate", "slow_read", "count", "echo"}
    finally:
        await mount.close()


# -- reconnect helpers ---------------------------------------------------------


def _pids(pattern: str) -> list[int]:
    """Pids whose full command line contains `pattern`, via `pgrep -f`."""
    proc = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(line) for line in proc.stdout.split() if line.strip()]


async def _wait_for_pid(pattern: str, *, timeout: float = 10.0) -> int:
    """The pid of the (single) running fixture matching `pattern`.

    Bounded and polled rather than slept: a freshly spawned subprocess is
    not necessarily visible to `pgrep` the instant `mount()` returns.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        found = _pids(pattern)
        if found:
            return found[-1]
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out after {timeout}s waiting for a process matching {pattern!r}")


async def _kill_and_wait(pid: int, *, timeout: float = 10.0) -> None:
    """`SIGKILL` a fixture and block until the kernel has actually reaped it.

    Returning early would let a test issue its next tool call while the
    server is still alive enough to answer it, which would silently test
    nothing.
    """
    os.kill(pid, signal.SIGKILL)
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:  # pragma: no cover -- zombie owned by us; treat as gone
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out after {timeout}s waiting for pid {pid} to die")


def _tool(tools: list[Tool], suffix: str) -> Tool:
    return next(t for t in tools if t.name.endswith(f"__{suffix}"))


@pytest.fixture(autouse=True)
def _instant_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reconnect backoff is real time. The tests here are about the decision
    logic, not about the wall-clock spacing, so the base delay is zeroed --
    otherwise the bounded-attempts test alone would add seconds to a suite
    that runs in well under a minute. `test_backoff_grows_between_attempts`
    below is the one that pins the spacing itself, and it opts back out.
    """
    monkeypatch.setattr(mcp_mount, "RECONNECT_BACKOFF_SECONDS", 0.0)


# -- reconnect -----------------------------------------------------------------


async def test_read_only_tool_retries_transparently_after_a_death(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only call is provably safe to retry, so a death mid-call is
    invisible to the caller."""
    name = "ro-retry"
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, name=name)])
    try:
        tools = await mount.mount()
        echo = _tool(tools, "echo")
        assert await echo.fn(text="before") == "before"
        await _kill_and_wait(await _wait_for_pid(f"mcp_flaky_server.py {name}"))
        # No error, no exception: the caller never learns the server died.
        assert await echo.fn(text="after") == "after"
        assert mount._servers[0].reconnect_count == 1
    finally:
        await mount.close()


async def test_mutating_tool_reports_unknown_outcome_rather_than_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The side effect may have landed. Reporting failure invites a retry
    that double-executes; the text must say the outcome is unknown."""
    name = "unknown-outcome"
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, name=name)])
    try:
        tools = await mount.mount()
        mutate = _tool(tools, "slow_mutate")
        await _kill_and_wait(await _wait_for_pid(f"mcp_flaky_server.py {name}"))
        with pytest.raises(RuntimeError) as excinfo:
            await mutate.fn(value="v")
        assert isinstance(excinfo.value, UnknownToolOutcome)
        message = str(excinfo.value)
        assert "UNKNOWN" in message
        assert "Do not retry" in message
        # The connection is dropped rather than rebuilt here: the rebuild
        # would only benefit later calls, and this raise must not wait for
        # it (see `test_mutating_death_raises_unknown_before_paying_for_the_
        # reconnect`). What matters is that the next call still gets a
        # working server, through the pre-dispatch branch.
        server = mount._servers[0]
        assert server.alive is False
        assert server.reconnect_count == 0
        assert await _tool(tools, "echo").fn(text="ok") == "ok"
        assert server.alive is True
        assert server.reconnect_count == 1
    finally:
        await mount.close()


async def test_mutating_death_raises_unknown_before_paying_for_the_reconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reconnect burst can outlast the router's own timeout, in which case
    the caller sees a timeout instead of the unknown-outcome message -- on the
    exact path that message exists for. The rebuild benefits only later calls,
    so the raise must not wait for it."""
    name = "raise-first"
    script = tmp_path / "flaky_copy.py"
    script.write_text(FIXTURE.read_text())
    # A burst that is guaranteed to be slow: the command is broken below so
    # every attempt fails, and the backoff between them is real time (this
    # opts back out of the module's zero-backoff fixture).
    monkeypatch.setattr(mcp_mount, "RECONNECT_BACKOFF_SECONDS", 5.0)
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, name=name, script=script)])
    try:
        tools = await mount.mount()
        mutate = _tool(tools, "slow_mutate")
        script.write_text("import sys\n\nsys.exit(1)\n")
        await _kill_and_wait(await _wait_for_pid(f"flaky_copy.py {name}"))
        # Stands in for the Router's own `asyncio.wait_for(tool.fn(...),
        # tool.timeout)` backstop, and is deliberately far shorter than the
        # burst above: if the raise waited for the rebuild, this budget would
        # expire first and the caller would get a bare TimeoutError -- which
        # is not a RuntimeError, so this would fail rather than pass quietly.
        #
        # Measured at 0.48s, so the budget itself is not close. The way this
        # test can go red without a regression is the other branch: if the
        # supervisor stands down on its own between the kill and this call,
        # the call finds no session and takes the PRE-dispatch rebuild
        # branch, where the 5s backoff above is paid legitimately and blows
        # the budget. That is a different code path from the one under test
        # -- read the traceback before believing the timing.
        with pytest.raises(RuntimeError) as excinfo:
            await asyncio.wait_for(mutate.fn(value="v"), 2.0)
        assert "UNKNOWN" in str(excinfo.value)
        assert mount._servers[0].reconnect_count == 0
    finally:
        await mount.close()


async def test_single_execution_across_a_mid_call_death(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One call means one execution, even through a reconnect. This is a
    consent invariant: one approval must mean one execution."""
    name = "single-exec"
    state = tmp_path / "state"
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, state=state, name=name, slow_seconds=10.0)])
    try:
        tools = await mount.mount()
        mutate = _tool(tools, "slow_mutate")
        pid = await _wait_for_pid(f"mcp_flaky_server.py {name}")
        call = asyncio.ensure_future(mutate.fn(value="once"))
        # The marker proves the side effect is already on disk, so the kill
        # lands strictly after the mutation and strictly before the reply.
        await _wait_for_started_marker(state, "once")
        await _kill_and_wait(pid)
        with pytest.raises(RuntimeError, match="UNKNOWN"):
            await call
        # A restarted server reading the same on-disk record is the only
        # witness that survives the death: exactly one completed mutation.
        assert await _tool(tools, "count").fn() == "1"
    finally:
        await mount.close()


async def test_concurrent_calls_produce_exactly_one_reconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-server lock means a batch of failing calls reconnects once."""
    name = "one-reconnect"
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, name=name)])
    try:
        tools = await mount.mount()
        echo = _tool(tools, "echo")
        await _kill_and_wait(await _wait_for_pid(f"mcp_flaky_server.py {name}"))
        results = await asyncio.gather(*(echo.fn(text=f"c{i}") for i in range(5)))
        assert results == [f"c{i}" for i in range(5)]
        server = mount._servers[0]
        assert server.reconnect_count == 1
        assert server.reconnect_attempts == 1
    finally:
        await mount.close()


async def test_reconnect_is_bounded_at_three_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server that cannot be brought back is retried a bounded number of
    times and then stays down for the process lifetime."""
    name = "bounded"
    script = tmp_path / "flaky_copy.py"
    script.write_text(FIXTURE.read_text())
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, name=name, script=script)])
    try:
        tools = await mount.mount()
        echo = _tool(tools, "echo")
        assert await echo.fn(text="up") == "up"
        # Break the command before killing it, so every restart attempt fails.
        script.write_text("import sys\n\nsys.exit(1)\n")
        await _kill_and_wait(await _wait_for_pid(f"flaky_copy.py {name}"))
        with pytest.raises(RuntimeError):
            await echo.fn(text="down")
        server = mount._servers[0]
        assert server.reconnect_attempts == mcp_mount.MAX_RECONNECT_ATTEMPTS == 3
        assert server.alive is False
        # Retired: a later call must not start the whole dance over again.
        with pytest.raises(RuntimeError):
            await echo.fn(text="down again")
        assert server.reconnect_attempts == 3
    finally:
        await mount.close()


async def test_configured_reconnect_attempts_bounds_the_burst(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mount configured with a smaller budget than the module default spends
    exactly that budget: one attempt per burst, not three."""
    name = "one-attempt"
    script = tmp_path / "flaky_copy.py"
    script.write_text(FIXTURE.read_text())
    mount = MCPMount(
        [flaky_cmd(tmp_path, monkeypatch, name=name, script=script)], reconnect_attempts=1
    )
    try:
        tools = await mount.mount()
        echo = _tool(tools, "echo")
        assert await echo.fn(text="up") == "up"
        script.write_text("import sys\n\nsys.exit(1)\n")
        await _kill_and_wait(await _wait_for_pid(f"flaky_copy.py {name}"))
        with pytest.raises(RuntimeError):
            await echo.fn(text="down")
        server = mount._servers[0]
        assert server.reconnect_attempts == 1
        assert server.reconnect_count == 1
        assert server.alive is False
    finally:
        await mount.close()


async def test_configured_backoff_and_cooldown_replace_the_module_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other two configured values, measured against a stubbed bring-up so
    the assertions are about the budget arithmetic alone. A configured value
    wins over the module constant; what is NOT configured still falls back to
    it (the autouse fixture's zeroed backoff is what the second burst below
    would otherwise pay)."""
    delays: list[float] = []

    async def fake_sleep(seconds: float, *args: Any, **kwargs: Any) -> Any:
        delays.append(seconds)
        return await asyncio.sleep(0, *args, **kwargs)

    class _AsyncioProxy:
        sleep = staticmethod(fake_sleep)

        def __getattr__(self, name: str) -> Any:
            return getattr(asyncio, name)

    async def always_fails(op: str) -> BaseException | None:
        return RuntimeError("nope")

    server = _MountedServer(
        MCPServerConfig(command="unused"),
        reconnect_attempts=2,
        reconnect_backoff_seconds=0.25,
        reconnect_cooldown_seconds=30.0,
    )
    server._mounted = True
    server.alive = False
    monkeypatch.setattr(mcp_mount, "asyncio", _AsyncioProxy())
    monkeypatch.setattr(server, "_submit", always_fails)

    assert await server.reconnect(server.generation) is False
    # Two attempts, one gap between them, at the configured base delay.
    assert server.reconnect_attempts == 2
    assert delays == [0.25]
    # And the burst that just failed starts the configured cooldown, so the
    # next caller fails fast without spending a single further attempt.
    assert await server.reconnect(server.generation) is False
    assert server.reconnect_attempts == 2
    assert server._cooldown_until - asyncio.get_running_loop().time() > 25.0


async def test_backoff_grows_between_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retries are spaced, and the spacing grows. Measured against a
    stubbed bring-up so the assertion is about the delays alone and costs
    no subprocess spawns."""
    monkeypatch.setattr(mcp_mount, "RECONNECT_BACKOFF_SECONDS", 0.02)
    server = _MountedServer(MCPServerConfig(command="unused"))
    server._mounted = True
    server.alive = False
    delays: list[float] = []

    async def fake_sleep(seconds: float, *args: Any, **kwargs: Any) -> Any:
        delays.append(seconds)
        return await asyncio.sleep(0, *args, **kwargs)

    async def always_fails(op: str) -> BaseException | None:
        return RuntimeError("nope")

    class _AsyncioProxy:
        """Everything `asyncio` has, except `sleep` records its argument and
        returns immediately. Swapped in for the module's OWN `asyncio` name
        rather than patching `asyncio.sleep` itself, so no other code in the
        process sees a stubbed clock."""

        sleep = staticmethod(fake_sleep)

        def __getattr__(self, name: str) -> Any:
            return getattr(asyncio, name)

    monkeypatch.setattr(mcp_mount, "asyncio", _AsyncioProxy())
    monkeypatch.setattr(server, "_submit", always_fails)
    assert await server.reconnect(server.generation) is False
    # Three attempts, two gaps between them, each double the last.
    assert delays == [0.02, 0.04]


async def test_server_that_failed_to_mount_is_never_reconnected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A startup failure stays skipped for the process lifetime."""
    script = tmp_path / "never_starts.py"
    script.write_text("import sys\n\nsys.exit(1)\n")
    mount = MCPMount([shlex.join([sys.executable, str(script)])])
    try:
        assert await mount.mount() == []
        # Never registered, so nothing can ever ask it to reconnect.
        assert mount._servers == []
    finally:
        await mount.close()

    # And the object itself refuses, without building any transport at all.
    server = _MountedServer(MCPServerConfig(command=shlex.join([sys.executable, str(script)])))
    assert await server.reconnect(server.generation) is False
    assert server.reconnect_attempts == 0
    assert server.session is None
    await server.close()


async def test_annotation_drift_force_gates_the_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool that comes back with read_only_hint flipped is force-gated to
    confirm=True with a warning -- the registered classification is no longer
    something the server can be trusted to have kept."""
    name = "drifter"
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, name=name)])
    try:
        tools = await mount.mount()
        echo = _tool(tools, "echo")
        assert echo.read_only is True
        assert echo.confirm is False
        # The restarted incarnation reads this at import time.
        monkeypatch.setenv("MCP_FLAKY_ANNOTATIONS", "drift")
        await _kill_and_wait(await _wait_for_pid(f"mcp_flaky_server.py {name}"))
        # count() is read-only, so this call reconnects and retries cleanly;
        # the drift is noticed as part of that reconnect.
        assert await _tool(tools, "count").fn() == "0"
        assert echo.confirm is True
        assert echo.read_only is False
    finally:
        await mount.close()


async def test_drifted_tool_is_no_longer_retried_transparently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Losing read-only status must also lose the transparent retry that
    read-only status is what justified."""
    name = "drift-no-retry"
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, name=name)])
    try:
        tools = await mount.mount()
        echo = _tool(tools, "echo")
        monkeypatch.setenv("MCP_FLAKY_ANNOTATIONS", "drift")
        await _kill_and_wait(await _wait_for_pid(f"mcp_flaky_server.py {name}"))
        assert await _tool(tools, "count").fn() == "0"
        assert echo.read_only is False
        # Kill again: echo is no longer trusted read-only, so the death is
        # reported as an unknown outcome instead of being retried away.
        await _kill_and_wait(await _wait_for_pid(f"mcp_flaky_server.py {name}"))
        with pytest.raises(RuntimeError, match="UNKNOWN"):
            await echo.fn(text="x")
    finally:
        await mount.close()


async def test_drift_discovered_by_a_call_s_own_reconnect_stops_that_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry decision cannot be made before the reconnect that reveals
    the drift. `echo` is the tool that triggers the reconnect here, so its
    new, MUTATING annotation is discovered by that very reconnect -- and is
    first knowable only after it. Retrying anyway would execute, with no
    confirm gate and no approval, a call the server has just declared
    destructive.

    Deliberately unlike the two drift tests above, which spend the
    drift-discovering reconnect on `count()` and only then touch `echo`:
    that ordering records the drift before the call under test, so it cannot
    reach this case at all.
    """
    name = "drift-self"
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, name=name)])
    try:
        tools = await mount.mount()
        echo = _tool(tools, "echo")
        assert echo.read_only is True
        monkeypatch.setenv("MCP_FLAKY_ANNOTATIONS", "drift")
        await _kill_and_wait(await _wait_for_pid(f"mcp_flaky_server.py {name}"))
        # First call after the death, and the first to see the restarted
        # server's annotations: no transparent retry, an unknown outcome.
        with pytest.raises(RuntimeError, match="UNKNOWN"):
            await echo.fn(text="x")
        # The reconnect still happened and still applied the drift -- what
        # is withheld is only the re-invocation.
        assert mount._servers[0].reconnect_count == 1
        assert echo.read_only is False and echo.confirm is True
    finally:
        await mount.close()


async def test_vanished_tool_returns_an_error_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool the restarted server no longer advertises cannot be called;
    its next invocation is an error result, not a call into the void."""
    name = "vanisher"
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, name=name)])
    try:
        tools = await mount.mount()
        echo = _tool(tools, "echo")
        monkeypatch.setenv("MCP_FLAKY_DROP", "echo")
        await _kill_and_wait(await _wait_for_pid(f"mcp_flaky_server.py {name}"))
        assert await _tool(tools, "count").fn() == "0"
        with pytest.raises(RuntimeError, match="no longer advertises"):
            await echo.fn(text="gone")
        # Siblings that did come back are unaffected.
        assert await _tool(tools, "count").fn() == "0"
    finally:
        await mount.close()


class _PoisonedSession:
    """A `ClientSession` stand-in for the state the SDK leaves behind after a
    transport death: its cancel scope is cancelled, so every operation on it
    raises a bare `CancelledError` rather than an `MCPError`.

    Injected rather than provoked, because provoking it reliably from a real
    subprocess death is not something these tests can arrange: which of the
    two exceptions a caller sees depends on internal SDK timing rather than
    on anything the test controls. The branch is what matters here, so it is
    driven directly -- everything around the injection (the mount, the
    `Tool.fn`, the reconnect it triggers) is real.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise asyncio.CancelledError("Cancelled via cancel scope 0xdead")


async def test_cancelled_error_never_escapes_a_tool_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A BaseException escaping Tool.fn would poison the Router's task."""
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, name="poisoned")])
    try:
        tools = await mount.mount()
        server = mount._servers[0]

        # Read-only: swallowed, reconnected, retried, invisible to the caller.
        poisoned = _PoisonedSession()
        server.session = poisoned  # type: ignore[assignment]
        assert await _tool(tools, "echo").fn(text="hi") == "hi"
        assert poisoned.calls == 1
        assert server.reconnect_count == 1

        # Mutating: still no BaseException, just an ordinary error result.
        server.session = _PoisonedSession()  # type: ignore[assignment]
        with pytest.raises(RuntimeError) as excinfo:
            await _tool(tools, "slow_mutate").fn(value="v")
        # An ordinary Exception, whatever its exact class -- what must not
        # come out is anything that only `except BaseException` would catch.
        assert isinstance(excinfo.value, Exception)
        assert not isinstance(excinfo.value, asyncio.CancelledError)
    finally:
        await mount.close()


async def test_router_timeout_still_surfaces_as_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Router runs every tool under `asyncio.wait_for`. Swallowing the
    cancellation that a timeout delivers would turn a slow tool into an
    "unknown outcome" error, so a self-requested cancellation is passed
    through untouched while an unprompted one is not."""
    mount = MCPMount(
        [flaky_cmd(tmp_path, monkeypatch, name="timeout-passthrough", slow_seconds=1.0)]
    )
    try:
        tools = await mount.mount()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(_tool(tools, "slow_mutate").fn(value="slow"), 0.2)
        assert mount._servers[0].reconnect_count == 0
    finally:
        await mount.close()


async def test_call_timeout_reaches_the_caller_as_the_coded_native_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of enforcing call_timeout through the SDK's own
    `read_timeout_seconds` instead of a Router-level `asyncio.wait_for`: a
    genuinely slow tool call must surface a coded `MCPError(REQUEST_TIMEOUT)`
    -- which `_is_connection_death` correctly reads as "not a dead
    connection" -- rather than the Router's bare, uncoded `TimeoutError`.

    Wrapped in `asyncio.wait_for(tool.fn(...), tool.timeout)`, exactly the
    way `router.py` itself calls it, so this exercises the real race between
    the two layers rather than asserting on a configured value in isolation.
    It would fail if `Tool.timeout` were ever set equal to (rather than
    materially larger than) the native call_timeout: the Router's outer
    clock starts before `tool.fn` even runs, so an equal budget lets it
    expire first and this test would see a bare `TimeoutError` instead.

    Driven through the READ-ONLY `slow_read`, because that is the tool whose
    timeout still reaches the caller as the raw `MCPError`. On a mutating
    tool the same coded error is re-raised as an unknown outcome before it
    gets out (see `test_mutating_timeout_reports_an_unknown_outcome`, which
    asserts the same code on the `__cause__`); the race between the two
    clocks is identical either way.
    """
    mount = MCPMount(
        [flaky_cmd(tmp_path, monkeypatch, name="native-timeout", slow_seconds=5.0)],
        call_timeout=0.2,
    )
    try:
        tools = await mount.mount()
        slow = _tool(tools, "slow_read")
        assert slow.timeout == pytest.approx(0.2 + ROUTER_TIMEOUT_MARGIN_SECONDS)
        with pytest.raises(MCPError) as exc_info:
            await asyncio.wait_for(slow.fn(), slow.timeout)
        assert exc_info.value.code == types.REQUEST_TIMEOUT
    finally:
        await mount.close()


async def test_mutating_timeout_reports_an_unknown_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mutating call that times out was already dispatched -- the server may
    complete it a second later. That is the same indeterminacy as a mid-call
    connection death, and it needs the same wording: reporting it as a plain
    failure invites a retry that performs the side effect twice."""
    mount = MCPMount(
        [flaky_cmd(tmp_path, monkeypatch, name="mutating-timeout", slow_seconds=5.0)],
        call_timeout=0.2,
    )
    try:
        tools = await mount.mount()
        slow = _tool(tools, "slow_mutate")
        # Called exactly the way router.py calls it, so the native timeout
        # and the Router's backstop race for real.
        with pytest.raises(RuntimeError) as exc_info:
            await asyncio.wait_for(slow.fn(value="v"), slow.timeout)
        # The TYPE is what the router keys on to avoid announcing this as a
        # failure, so it is part of the contract, not an implementation detail.
        assert isinstance(exc_info.value, UnknownToolOutcome)
        message = str(exc_info.value)
        assert "UNKNOWN" in message
        assert "Do not retry" in message
        # Only the wording changed: underneath it is still the coded native
        # timeout, and a timeout is still not a dead transport.
        cause = exc_info.value.__cause__
        assert isinstance(cause, MCPError) and cause.code == types.REQUEST_TIMEOUT
        assert mount._servers[0].reconnect_count == 0
        assert mount._servers[0].alive is True
    finally:
        await mount.close()


async def test_read_only_timeout_is_still_reported_as_a_plain_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only call has no side effect to be ambiguous about, so it must
    not inherit the unknown-outcome wording -- that would train the model to
    treat every slow read as dangerous."""
    mount = MCPMount(
        [flaky_cmd(tmp_path, monkeypatch, name="read-only-timeout", slow_seconds=5.0)],
        call_timeout=0.2,
    )
    try:
        tools = await mount.mount()
        slow = _tool(tools, "slow_read")
        # The raised type is itself the assertion: an `UnknownToolOutcome`
        # would not be an `MCPError` and would fail here before the wording
        # is even looked at.
        with pytest.raises(MCPError) as exc_info:
            await asyncio.wait_for(slow.fn(), slow.timeout)
        message = str(exc_info.value)
        assert "timed out" in message.lower()
        assert "UNKNOWN" not in message
        assert "Do not retry" not in message
        assert mount._servers[0].reconnect_count == 0
    finally:
        await mount.close()


# -- HTTP transport ------------------------------------------------------------
#
# The stdio tests above cannot reach one whole class of failure: over stdio the
# client library owns the server process, so "the connection died" and "the
# process is being respawned" are the same event. An HTTP server is a separate
# thing that can be absent for a while and then come back, which is what these
# two cover. Like the rest of this module they are unmarked and run every
# time -- each starts a real ASGI server, but that cost is small next to what
# these two prove and nothing here should be dropped by a fast local lane.


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_http_fixture(
    port: int, state: Path, *, slow_seconds: float, log: Path
) -> subprocess.Popen[bytes]:
    """Start the fixture in HTTP mode, keeping its output.

    The output goes to a file rather than to `DEVNULL` so that a failure to
    bind or import shows up as a diagnostic instead of only as a timeout
    waiting for a port that is never going to open.
    """
    env = dict(
        os.environ,
        MCP_FLAKY_HTTP_PORT=str(port),
        MCP_FLAKY_STATE=str(state),
        MCP_FLAKY_SLOW_SECONDS=str(slow_seconds),
    )
    handle = log.open("ab")
    try:
        return subprocess.Popen(
            [sys.executable, str(FIXTURE), "httpflaky"],
            env=env,
            stdout=handle,
            stderr=handle,
        )
    finally:
        handle.close()


async def _wait_for_port(port: int, *, listening: bool, log: Path, timeout: float = 20.0) -> None:
    """Block until the port is (or is no longer) accepting connections.

    Polled rather than slept: an ASGI server's startup time is not something
    a test can assume, and guessing would make these race. On timeout the
    server's own output is included, which is usually the actual answer.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.2)
            open_now = sock.connect_ex(("127.0.0.1", port)) == 0
        if open_now == listening:
            return
        await asyncio.sleep(0.05)
    output = log.read_text(errors="replace") if log.exists() else "<no output captured>"
    raise AssertionError(
        f"timed out waiting for port {port} listening={listening}; server output:\n{output}"
    )


async def test_http_server_restart_reconnects_transparently(tmp_path: Path) -> None:
    """An HTTP server that goes away and comes back is still reachable.

    The restarted server is perfectly healthy -- it simply has no memory of
    the session id this client holds, and rejects it. That has to count as a
    dead connection, or every restart of a remote MCP server would silently
    disable its tools until dudamel itself was restarted.
    """
    port = _free_port()
    state = tmp_path / "state"
    log = tmp_path / "server.log"
    server_proc = _start_http_fixture(port, state, slow_seconds=0.05, log=log)
    mount = MCPMount([MCPServerConfig(url=f"http://127.0.0.1:{port}/mcp")])
    try:
        await _wait_for_port(port, listening=True, log=log)
        tools = await mount.mount()
        echo = _tool(tools, "echo")
        assert await echo.fn(text="before") == "before"

        server_proc.kill()
        server_proc.wait()
        await _wait_for_port(port, listening=False, log=log)
        server_proc = _start_http_fixture(port, state, slow_seconds=0.05, log=log)
        await _wait_for_port(port, listening=True, log=log)

        # No error surfaces: the stale session is recognized, rebuilt, and
        # the read-only call retried against the new one.
        assert await echo.fn(text="after") == "after"
        assert mount._servers[0].reconnect_count == 1
    finally:
        await mount.close()
        server_proc.kill()
        server_proc.wait()


async def test_http_background_failure_does_not_strand_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transport that fails in the background cancels the task that hosts
    its cancel scopes -- the supervisor.

    Left unhandled, that killed the supervisor with the connection's stack
    still open: nothing ever released it, and every later request found no
    supervisor to talk to, so the server was unreachable for the rest of the
    process with no way back. What is asserted here is that outcome, not the
    mechanism: whether the supervisor rides the cancellation out or stands
    down and is replaced, the server has to come back.
    """
    port = _free_port()
    state = tmp_path / "state"
    log = tmp_path / "server.log"
    # Belt and braces: the asserted path below spends no reconnect budget at
    # all, but a burst from anywhere else must not park this test behind a
    # 60-second cooldown.
    monkeypatch.setattr(mcp_mount, "RECONNECT_COOLDOWN_SECONDS", 0.2)
    server_proc = _start_http_fixture(port, state, slow_seconds=30.0, log=log)
    mount = MCPMount([MCPServerConfig(url=f"http://127.0.0.1:{port}/mcp")])
    call: asyncio.Future[str] | None = None
    try:
        await _wait_for_port(port, listening=True, log=log)
        tools = await mount.mount()
        server = mount._servers[0]
        call = asyncio.ensure_future(_tool(tools, "slow_mutate").fn(value="inflight"))
        await _wait_for_started_marker(state, "inflight", timeout=20.0)

        server_proc.kill()
        server_proc.wait()
        await _wait_for_port(port, listening=False, log=log)

        # Mutating, and the request was in flight, so the outcome is unknown.
        with pytest.raises(RuntimeError, match="UNKNOWN"):
            await call
        assert server.session is None
        # No reconnect burst was started for it, and so no cooldown was
        # spent: a mutating call reports the unknown outcome and leaves the
        # rebuild to whoever calls next.
        assert server.reconnect_count == 0

        # The server comes back, as a restarted one eventually does. Nothing
        # is stranded: the connection rebuilds -- inside the read-only call's
        # own pre-dispatch branch -- and the tools work again.
        server_proc = _start_http_fixture(port, state, slow_seconds=0.05, log=log)
        await _wait_for_port(port, listening=True, log=log)
        assert await _tool(tools, "echo").fn(text="recovered") == "recovered"
        assert server.alive is True
    finally:
        # A pending call whose exception is never retrieved would otherwise
        # surface as an unraisable-exception warning if an assertion above
        # failed before it was awaited.
        if call is not None and not call.done():
            call.cancel()
        await mount.close()
        server_proc.kill()
        server_proc.wait()


# -- the classifier, on its own ------------------------------------------------


@pytest.mark.parametrize(
    ("error", "is_death"),
    [
        (MCPError(code=types.CONNECTION_CLOSED, message="Connection closed"), True),
        (MCPError(code=types.CONNECTION_CLOSED, message="SSE stream ended"), True),
        # A restarted HTTP server rejecting the session id it no longer knows.
        (MCPError(code=types.INVALID_REQUEST, message="Session terminated"), True),
        (MCPError(code=types.INVALID_REQUEST, message="Session not found"), True),
        (MCPError(code=types.INVALID_REQUEST, message="session expired"), True),
        (asyncio.CancelledError("via cancel scope"), True),
        (BaseExceptionGroup("g", [MCPError(code=types.CONNECTION_CLOSED, message="x")]), True),
        # A slow tool is not a dead transport.
        (MCPError(code=types.REQUEST_TIMEOUT, message="Request timed out"), False),
        # Ordinary protocol errors must not trigger connection churn...
        (MCPError(code=types.INVALID_REQUEST, message="Missing required field"), False),
        (MCPError(code=types.INVALID_PARAMS, message="bad params"), False),
        # ...including one that merely mentions a session.
        (MCPError(code=types.INVALID_REQUEST, message="session not found in workspace"), False),
        (RuntimeError("tool blew up"), False),
        (BaseExceptionGroup("g", [RuntimeError("unrelated")]), False),
    ],
)
def test_connection_death_classification(error: BaseException, is_death: bool) -> None:
    """The classifier gates the whole retry decision, so it is pinned here
    directly rather than only through tests that need a live server."""
    assert _is_connection_death(error) is is_death


async def test_marking_a_connection_lost_never_clobbers_a_newer_one() -> None:
    """Router tool calls in one batch run concurrently, and a batch of
    simultaneously-failing calls is exactly what `reconnect()` coalesces. A
    caller that diagnosed connection N as dead must not be able to clear a
    connection some sibling has already rebuilt: a sibling that then reads
    `server.session` would be told there is no live session about a
    connection that is up, and would rebuild it again for nothing.

    Driven directly rather than through two racing subprocesses -- the
    window is a few statements wide and could not be hit reliably, but the
    generation check that closes it is exact.
    """
    server = _MountedServer(MCPServerConfig(command="true"))
    server.generation = 4
    server.alive = True
    server.session = object()  # type: ignore[assignment]

    server.mark_connection_lost(3)  # a sibling already rebuilt past this
    assert server.alive is True and server.session is not None

    server.mark_connection_lost(4)  # the caller's own connection
    assert server.alive is False and server.session is None


# -- supervisor stand-down -----------------------------------------------------


async def _wait_done(task: asyncio.Task[Any], *, timeout: float = 5.0) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if task.done():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("supervisor did not finish after being cancelled")


async def test_cancelling_the_supervisor_lets_it_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled supervisor must release its stack AND complete.

    Absorbing the cancellation and going back to waiting would make the task
    effectively uncancellable, which is what hangs an event loop's shutdown.
    """
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, name="standdown")])
    try:
        await mount.mount()
        server = mount._servers[0]
        supervisor = server._supervisor
        assert supervisor is not None
        supervisor.cancel()
        await _wait_done(supervisor)
        assert server.session is None
        assert server._stack is None
        # ...and the server is not stranded: the next call gets a new one.
        assert await server.reconnect(server.generation) is True
        assert server.alive is True
    finally:
        await mount.close()


async def test_repeated_supervisor_cancellation_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stand-downs without a working connection in between are not
    unlimited: something cancelling the supervisor in a loop must not make
    every tool call respawn one forever."""
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, name="standdown-bound")])
    try:
        await mount.mount()
        server = mount._servers[0]

        async def never_connects(stack: Any) -> None:
            raise RuntimeError("cannot connect")

        # No successful connection can intervene, so the stand-downs count.
        monkeypatch.setattr(server, "_connect", never_connects)
        for _ in range(mcp_mount.MAX_SCOPE_RECOVERIES + 1):
            supervisor = server._supervisor
            assert supervisor is not None
            supervisor.cancel()
            await _wait_done(supervisor)
            await server._submit("connect")  # respawns, and fails to connect
        assert server._supervisor_failed is True
        assert await server.reconnect(server.generation) is False
        assert isinstance(await server._submit("connect"), BaseException)
    finally:
        await mount.close()


SRC_DIR = Path(__file__).parent.parent / "src"


def _run_shutdown_script(script: Path, body: str) -> None:
    """Run `body` under `asyncio.run` in a fresh interpreter and require a
    clean exit.

    A subprocess because the thing under test is `asyncio.run`'s own
    shutdown sequence, which cannot be exercised from inside a running event
    loop. A hang shows up as a timeout, failing the test instead of wedging
    the suite.

    Started in its own session and torn down with a process-GROUP kill. When
    this test fails it fails by hanging, and the hung interpreter is itself
    the parent of an MCP server subprocess; killing only the direct child
    would orphan that grandchild to linger for the length of its own sleep.
    A red run must not leave strays behind on the machine.
    """
    script.write_text(
        "import asyncio, sys\n"
        f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
        "from dudamel.mcp_mount import MCPMount\n"
        "async def main():\n"
        f"{body}"
        "asyncio.run(main())\n"
        "print('EXITED CLEANLY')\n"
    )
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # makes `proc` a process-group leader
    )
    try:
        try:
            stdout, stderr = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            raise AssertionError(
                f"{script.name} did not exit within 60s -- loop shutdown hung"
            ) from None
        assert "EXITED CLEANLY" in stdout, f"stdout={stdout!r} stderr={stderr[-2000:]!r}"
        assert proc.returncode == 0, f"stderr={stderr[-2000:]!r}"
    finally:
        # Unconditionally, not just on a hang. Even a clean exit can leave
        # the MCP server subprocess behind -- the interpreter is shutting
        # down mid-handshake, so the transport does not always get to reap
        # its child -- and an orphan would then idle for as long as its own
        # sleep. Signalling the whole group catches those; ProcessLookupError
        # just means there was nothing left to kill.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def test_an_unclosed_mount_does_not_hang_loop_shutdown(tmp_path: Path) -> None:
    """`asyncio.run` cancels each surviving task exactly once and then waits
    for it. A supervisor that absorbed that cancellation and went back to
    waiting would never finish, so the interpreter would hang on exit
    instead of reporting whatever the real problem was.

    This is the idle case: the supervisor is parked on its request queue.
    """
    _run_shutdown_script(
        tmp_path / "unclosed_idle.py",
        f"    mount = MCPMount([{shlex.join([sys.executable, str(FIXTURE)])!r}])\n"
        "    assert await mount.mount()\n"
        "    # deliberately never closed\n",
    )


def test_an_unclosed_mount_does_not_hang_with_a_connect_in_flight(tmp_path: Path) -> None:
    """The same shutdown, but with the supervisor inside a connection attempt
    rather than idle.

    This is the harder half. A `CancelledError` swallowed inside `_rebuild`
    -- returned as a failed attempt instead of re-raised -- puts the
    supervisor back on its idle await holding a cancellation that was meant
    to stop it, and the process never exits. Unlike the idle case there is
    no second cancellation to rescue it.

    The script mounts successfully first and then hangs a *reconnect*,
    rather than hanging the initial mount. That is deliberate: a hung
    initial mount ends up in `mount()`'s own `except BaseException`, which
    calls `close()` and stops the supervisor as a side effect -- so the
    process exits for a reason that has nothing to do with the line under
    test, and the test silently stops testing it. `reconnect()` has no such
    handler, so nothing but the supervisor's own stand-down can end it.
    """
    _run_shutdown_script(
        tmp_path / "unclosed_connecting.py",
        f"    mount = MCPMount([{shlex.join([sys.executable, str(FIXTURE)])!r}])\n"
        "    assert await mount.mount()\n"
        "    server = mount._servers[0]\n"
        "    async def never_finishes(stack):\n"
        "        await asyncio.sleep(3600)\n"
        "    server._connect = never_finishes\n"
        "    server.alive = False\n"
        "    server.session = None\n"
        "    reconnecting = asyncio.create_task(server.reconnect(server.generation))\n"
        "    await asyncio.sleep(0.5)  # let the supervisor park inside _rebuild\n"
        "    assert not reconnecting.done()\n"
        "    # deliberately never closed, with a connect still in flight\n",
    )


# -- the reconnect budget bounds a burst, not the process ----------------------


async def test_reconnect_budget_rearms_after_the_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server that is still down when its burst runs out is not written
    off. Real restarts routinely take longer than one burst, so after the
    cooldown the next call gets a fresh burst -- and succeeds if the server
    came back in the meantime."""
    name = "rearm"
    script = tmp_path / "rearm_fixture.py"
    original = FIXTURE.read_text()
    script.write_text(original)
    monkeypatch.setattr(mcp_mount, "RECONNECT_COOLDOWN_SECONDS", 0.2)
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, name=name, script=script)])
    try:
        tools = await mount.mount()
        echo = _tool(tools, "echo")
        server = mount._servers[0]

        # Server is unrestartable, so the first burst fails outright.
        script.write_text("import sys\n\nsys.exit(1)\n")
        await _kill_and_wait(await _wait_for_pid(f"rearm_fixture.py {name}"))
        with pytest.raises(RuntimeError):
            await echo.fn(text="down")
        assert server.reconnect_attempts == mcp_mount.MAX_RECONNECT_ATTEMPTS
        assert server.alive is False

        # Inside the cooldown: fails fast, no new attempts at all.
        with pytest.raises(RuntimeError):
            await echo.fn(text="still down")
        assert server.reconnect_attempts == mcp_mount.MAX_RECONNECT_ATTEMPTS

        # The server comes back, the cooldown expires, and so does the
        # write-off: the next call reconnects instead of failing forever.
        script.write_text(original)
        await asyncio.sleep(0.25)
        assert await echo.fn(text="back") == "back"
        assert server.alive is True
        assert server.reconnect_attempts == mcp_mount.MAX_RECONNECT_ATTEMPTS + 1
    finally:
        await mount.close()


async def test_cancelling_connect_attempts_do_not_retire_a_server_across_bursts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server that stays down for several bursts must still come back.

    Measured against a fully-down HTTP server: every attempt in a burst ends
    in a cancellation from the connection it was building, which stands the
    supervisor down. That is reproduced here by making the connect itself
    raise, because it is the accounting that matters, not the transport. If
    those stand-downs accumulated across bursts they would exhaust
    `MAX_SCOPE_RECOVERIES` after two or three cooldowns and write the server
    off permanently -- which is precisely what the cooldown exists to
    prevent, arrived at by a slower route.
    """
    monkeypatch.setattr(mcp_mount, "RECONNECT_COOLDOWN_SECONDS", 0.05)
    mount = MCPMount([flaky_cmd(tmp_path, monkeypatch, name="across-bursts")])
    try:
        tools = await mount.mount()
        echo = _tool(tools, "echo")
        server = mount._servers[0]
        healthy_connect = server._connect

        async def cancelled_connect(stack: Any) -> None:
            raise asyncio.CancelledError("Cancelled via cancel scope")

        monkeypatch.setattr(server, "_connect", cancelled_connect)
        server.alive = False
        server.session = None

        # Four full bursts' worth of downtime -- more than the budget could
        # absorb if it carried over.
        for _ in range(4):
            with pytest.raises(RuntimeError):
                await echo.fn(text="down")
            assert server._standdowns <= mcp_mount.MAX_SCOPE_RECOVERIES
            await asyncio.sleep(0.06)  # let the cooldown lapse
        assert server._supervisor_failed is False

        # The server comes back after all that, and so do its tools.
        monkeypatch.setattr(server, "_connect", healthy_connect)
        assert await echo.fn(text="back") == "back"
        assert server.alive is True
    finally:
        await mount.close()
