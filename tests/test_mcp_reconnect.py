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
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from dudamel import mcp_mount
from dudamel.contract.types import Tool
from dudamel.mcp_mount import MCPMount, MCPServerConfig, _MountedServer

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
        message = str(excinfo.value)
        assert "UNKNOWN" in message
        assert "Do not retry" in message
        # The server itself is back up even though this call was not retried.
        assert mount._servers[0].alive is True
        assert await _tool(tools, "echo").fn(text="ok") == "ok"
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
    monkeypatch.setattr(mcp_mount, "MOUNT_TIMEOUT", 5.0)
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
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float, *args: Any, **kwargs: Any) -> Any:
        delays.append(seconds)
        return await real_sleep(0, *args, **kwargs)

    async def always_fails(op: str) -> BaseException | None:
        return RuntimeError("nope")

    # Records the requested delay and yields immediately instead of waiting.
    # This replaces `asyncio.sleep` itself, so it still yields to the loop
    # for any other caller -- nothing else runs during this test, and
    # monkeypatch puts the real one back afterwards.
    monkeypatch.setattr(mcp_mount.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(server, "_submit", always_fails)
    assert await server.reconnect(server.generation) is False
    # Three attempts, two gaps between them, each double the last.
    assert delays == [0.02, 0.04]


async def test_server_that_failed_to_mount_is_never_reconnected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A startup failure stays skipped for the process lifetime."""
    monkeypatch.setattr(mcp_mount, "MOUNT_TIMEOUT", 5.0)
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

    Injected rather than provoked. A real stdio death can surface either
    exception depending on whether the session's reader noticed EOF before
    the next request was written, so provoking this specific branch from a
    real subprocess would be a race; the branch itself is what matters.
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
        assert type(excinfo.value) is RuntimeError
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
