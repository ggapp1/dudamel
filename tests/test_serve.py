"""Acceptance tests for dudamel/serve.py: the single-process assembly. All
timing-sensitive assertions poll with a deadline -- never a bare
sleep-and-hope, which would make failures flaky instead of deterministic.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from sqlalchemy import select

from dudamel import App, Orchestrator, Runtime
from dudamel.config import Settings, TierConfig, WebConfig
from dudamel.db import Database
from dudamel.llm.testing import FakeProvider
from dudamel.models_core import JobRun
from dudamel.scheduler import JobScheduler
from dudamel.serve import _InstanceLock, _prepare_uvicorn, serve

TOKEN = "s3cr3t-serve-token"  # noqa: S105 — test fixture, not a real credential

# `dudamel/__init__.py` does `from dudamel.serve import serve`, which rebinds
# the *attribute* `dudamel.serve` from the submodule to that function --
# `import dudamel.serve as serve_module` would silently resolve to the same
# function (dotted `import ... as` traverses attributes, starting from the
# already-shadowed `dudamel` package). Going through `sys.modules` sidesteps
# that entirely and gets the real module object, which is what needs
# patching: `serve()`'s own globals hold its OWN `TelegramInterface`/
# `resolve_telegram_token` names (bound once, by value, at import time), not
# a live lookup back through `dudamel.interfaces.telegram`.
serve_module = sys.modules["dudamel.serve"]


def make_orc() -> Orchestrator:
    app = App("stats", description="d")

    @app.widget(title="Streak", renderer="markdown")
    async def streak() -> str:
        return "3 days"

    @app.job(interval_seconds=0)
    async def tick() -> None:
        pass

    return Orchestrator(apps=[app])


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/serve.db",
        data_dir=tmp_path,
        llm_tiers={"standard": TierConfig(provider="fake", model="f")},
        web=WebConfig(host="127.0.0.1", port=0),
    )


async def wait_for_port(settings: Settings, *, timeout: float = 5.0) -> int:
    """Poll `settings.web.port` (0 == "not yet bound") until `serve()`
    rewrites it with the real, OS-assigned port."""
    deadline = time.monotonic() + timeout
    while settings.web.port == 0:
        if time.monotonic() >= deadline:
            raise AssertionError(f"server did not bind within {timeout}s")
        await asyncio.sleep(0.01)
    return settings.web.port


async def poll_job_runs(db: Database, job_id: str, *, timeout: float = 5.0) -> list[JobRun]:
    deadline = time.monotonic() + timeout
    while True:
        async with db.session() as s:
            rows = list(
                (await s.execute(select(JobRun).where(JobRun.job_id == job_id))).scalars().all()
            )
        if rows:
            return rows
        if time.monotonic() >= deadline:
            raise AssertionError(f"expected a job_runs row for {job_id!r} within {timeout}s")
        await asyncio.sleep(0.02)


async def cancel_and_await(task: asyncio.Task[None]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.fixture(autouse=True)
def _no_telegram(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test here exercises the "no Telegram configured" path (Global
    Constraints: web always on, Telegram only if a token is present) --
    scrub any real token out of the environment so that stays true
    regardless of what the host shell happens to have set."""
    monkeypatch.delenv("DUDAMEL_TELEGRAM_TOKEN", raising=False)


# --- health + widgets over a real socket -------------------------------------


async def test_health_and_widgets_respond_over_real_localhost_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DUDAMEL_WEB_TOKEN", TOKEN)
    settings = make_settings(tmp_path)
    task = asyncio.create_task(
        serve(make_orc(), settings, providers={"standard": FakeProvider([])})
    )
    try:
        port = await wait_for_port(settings)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
            health = await c.get("/health")
            assert health.status_code == 200
            body = health.json()
            assert body["status"] == "ok"
            assert body["db"] is True

            widgets = await c.get("/api/widgets", headers={"Authorization": f"Bearer {TOKEN}"})
            assert widgets.status_code == 200
            assert {w["id"] for w in widgets.json()} == {"streak"}

            # /api/* still requires auth even over the real socket.
            unauthed = await c.get("/api/widgets")
            assert unauthed.status_code == 401
    finally:
        await cancel_and_await(task)


async def test_serve_logs_dashboard_url_and_telegram_status_on_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """M2: a successful boot must be visible without reading source -- the
    dashboard URL/token env var and whether Telegram came up are the two
    facts an operator needs immediately after starting `dudamel run`."""
    monkeypatch.setenv("DUDAMEL_WEB_TOKEN", TOKEN)
    caplog.set_level(logging.INFO, logger="dudamel.serve")
    settings = make_settings(tmp_path)
    task = asyncio.create_task(
        serve(make_orc(), settings, providers={"standard": FakeProvider([])})
    )
    try:
        port = await wait_for_port(settings)
        assert any(
            f"dudamel dashboard: http://127.0.0.1:{port}" in r.message
            and "DUDAMEL_WEB_TOKEN" in r.message
            for r in caplog.records
        )
        assert any(r.message == "telegram: disabled" for r in caplog.records)
    finally:
        await cancel_and_await(task)


async def test_serve_logs_telegram_enabled_when_it_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _FakeTelegram.instances.clear()
    monkeypatch.setattr(serve_module, "resolve_telegram_token", lambda settings: "fake-token")
    monkeypatch.setattr(serve_module, "TelegramInterface", _FakeTelegram)
    caplog.set_level(logging.INFO, logger="dudamel.serve")
    settings = make_settings(tmp_path)
    task = asyncio.create_task(
        serve(make_orc(), settings, providers={"standard": FakeProvider([])})
    )
    try:
        await wait_for_port(settings)
        assert any(r.message == "telegram: enabled" for r in caplog.records)
    finally:
        await cancel_and_await(task)


# --- instance lock: second serve() rejected ----------------------------------


async def test_second_serve_call_same_data_dir_raises_already_running(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    task = asyncio.create_task(
        serve(make_orc(), settings, providers={"standard": FakeProvider([])})
    )
    try:
        await wait_for_port(settings)
        other = make_settings(tmp_path)  # same data_dir -> same lockfile path
        with pytest.raises(RuntimeError, match="already running"):
            await serve(make_orc(), other, providers={"standard": FakeProvider([])})
        # the rejected attempt must not have touched the first instance's
        # binding -- still reachable on its original port.
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{settings.web.port}") as c:
            assert (await c.get("/health")).status_code == 200
    finally:
        await cancel_and_await(task)


# --- clean shutdown: cancellation ---------------------------------------------


async def test_cancelling_serve_task_shuts_down_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    disposed: list[Database] = []
    original_dispose = Database.dispose

    async def spy_dispose(self: Database) -> None:
        disposed.append(self)
        await original_dispose(self)

    monkeypatch.setattr(Database, "dispose", spy_dispose)

    settings = make_settings(tmp_path)
    lockfile = settings.data_dir / ".dudamel.lock"
    task = asyncio.create_task(
        serve(make_orc(), settings, providers={"standard": FakeProvider([])})
    )
    await wait_for_port(settings)
    assert lockfile.exists()

    # scheduler fired at least one interval job during the window.
    probe_db = Database(settings.database_url)
    try:
        rows = await poll_job_runs(probe_db, "stats.tick")
        assert rows[0].status == "ok"
    finally:
        await probe_db.dispose()

    disposed_before = len(disposed)
    task.cancel()
    await asyncio.wait_for(task, timeout=5.0)  # must COMPLETE, not raise

    assert not lockfile.exists()
    assert len(disposed) > disposed_before  # Runtime.stop() disposed its engine


# --- clean shutdown: second cancellation during scheduler drain (regression) ---


async def test_second_cancellation_during_scheduler_drain_still_runs_runtime_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: `scheduler.shutdown()` gained a real suspension point (a
    bounded drain of in-flight job tasks) and is the only step in the
    teardown sequence with no `CancelledError` guard. A second cancellation
    landing while that drain is in progress must not skip
    `_stop_quietly("runtime", runtime.stop())` -- which disposes the DB
    engine and closes every mounted MCP subprocess -- the way it would if
    the `CancelledError` were left to propagate out of `scheduler.shutdown()`
    uncaught.

    `JobScheduler.shutdown` is replaced so the hazard is reproduced
    deterministically instead of raced against real job timing: the
    replacement cancels the running `serve()` task itself, then awaits a
    bare `asyncio.sleep(0)` -- asyncio delivers the now-pending cancellation
    at that await, exactly like a real second SIGTERM/`task.cancel()`
    landing mid-drain.
    """
    disposed: list[Database] = []
    original_dispose = Database.dispose

    async def spy_dispose(self: Database) -> None:
        disposed.append(self)
        await original_dispose(self)

    monkeypatch.setattr(Database, "dispose", spy_dispose)

    async def cancel_again_mid_drain(self: JobScheduler) -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        await asyncio.sleep(0)  # the pending cancellation is delivered here

    monkeypatch.setattr(JobScheduler, "shutdown", cancel_again_mid_drain)

    settings = make_settings(tmp_path)
    lockfile = settings.data_dir / ".dudamel.lock"
    orc = Orchestrator(apps=[])  # no jobs -- isolates the teardown path itself
    task = asyncio.create_task(serve(orc, settings, providers={"standard": FakeProvider([])}))
    await wait_for_port(settings)

    disposed_before = len(disposed)
    task.cancel()  # first cancellation: unblocks stop_event.wait(), enters teardown
    await asyncio.wait_for(task, timeout=5.0)  # must COMPLETE, not raise

    assert not lockfile.exists()
    assert len(disposed) > disposed_before  # Runtime.stop() still ran despite the 2nd cancel


# --- clean shutdown: SIGTERM ---------------------------------------------------


async def test_sigterm_shuts_down_cleanly(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    lockfile = settings.data_dir / ".dudamel.lock"
    task = asyncio.create_task(
        serve(make_orc(), settings, providers={"standard": FakeProvider([])})
    )
    port = await wait_for_port(settings)

    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.wait_for(task, timeout=5.0)  # must COMPLETE, not raise

    assert not lockfile.exists()
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
        with pytest.raises(httpx.ConnectError):
            await c.get("/health", timeout=1.0)


# --- clean shutdown: double SIGTERM (regression) --------------------------------

_DOUBLE_SIGTERM_CHILD_SCRIPT = """
import asyncio
import sys
from pathlib import Path

from dudamel import App, Orchestrator
from dudamel.config import Settings, TierConfig, WebConfig
from dudamel.llm.testing import FakeProvider
from dudamel.serve import serve


def make_orc():
    app = App("stats", description="d")

    @app.widget(title="Streak", renderer="markdown")
    async def streak():
        return "3 days"

    return Orchestrator(apps=[app])


async def main() -> None:
    data_dir = Path(sys.argv[1])
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{data_dir}/serve.db",
        data_dir=data_dir,
        llm_tiers={"standard": TierConfig(provider="fake", model="f")},
        web=WebConfig(host="127.0.0.1", port=0),
    )
    task = asyncio.create_task(
        serve(make_orc(), settings, providers={"standard": FakeProvider([])})
    )
    while settings.web.port == 0:
        await asyncio.sleep(0.01)
    print("ready", flush=True)
    await task


asyncio.run(main())
"""


def test_double_sigterm_50ms_apart_does_not_hard_kill_mid_shutdown(tmp_path: Path) -> None:
    """Regression test (CRITICAL finding): signal handlers used to be torn
    down -- restoring default SIGTERM/SIGINT disposition -- BEFORE the
    ordered shutdown sequence ran, in the outer `finally`. A second SIGTERM
    landing in that window then fell through to the OS default action for
    SIGTERM: an immediate, un-catchable process kill, leaving the lockfile
    behind and skipping the rest of shutdown (including `Runtime.stop()`'s
    DB dispose). This has to run `serve()` in a real subprocess (mirroring
    the existing subprocess pattern used elsewhere in this file) -- the bug
    is about actual OS signal disposition, which no in-process `os.kill`
    test can observe, since a single interpreter shares one
    signal-handling thread no matter how many handlers are "installed" on
    top of it.
    """
    script = tmp_path / "_double_sigterm_child.py"
    script.write_text(_DOUBLE_SIGTERM_CHILD_SCRIPT)
    lockfile = tmp_path / ".dudamel.lock"

    proc = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        ready = proc.stdout.readline()
        assert ready.strip() == "ready"
        assert lockfile.exists()

        proc.send_signal(signal.SIGTERM)
        time.sleep(0.05)
        proc.send_signal(signal.SIGTERM)

        exit_code = proc.wait(timeout=5.0)
        assert exit_code == 0
        assert not lockfile.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)


# --- optional Telegram wiring --------------------------------------------------


class _FakeTelegram:
    """Stands in for `TelegramInterface` so this suite can exercise serve()'s
    "token configured" branch (construction, start/stop, bind_notify) without
    a real PTB Application making network calls."""

    instances: list[_FakeTelegram] = []

    def __init__(self, runtime: Runtime, settings: Settings) -> None:
        self.runtime = runtime
        self.settings = settings
        self.calls: list[str] = []
        _FakeTelegram.instances.append(self)

    async def start(self) -> None:
        self.calls.append("start")

    async def stop(self) -> None:
        self.calls.append("stop")

    async def notify(self, text: str) -> None:
        self.calls.append(f"notify:{text}")


async def test_telegram_started_and_bound_when_token_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeTelegram.instances.clear()
    monkeypatch.setattr(serve_module, "resolve_telegram_token", lambda settings: "fake-token")
    monkeypatch.setattr(serve_module, "TelegramInterface", _FakeTelegram)

    settings = make_settings(tmp_path)
    task = asyncio.create_task(
        serve(make_orc(), settings, providers={"standard": FakeProvider([])})
    )
    await wait_for_port(settings)

    assert len(_FakeTelegram.instances) == 1
    fake = _FakeTelegram.instances[0]
    assert fake.calls == ["start"]

    # bind_notify(telegram.notify) rewired every app's app.notify() -- confirm
    # against the very Runtime instance serve() built (handed to the fake at
    # construction), not a fresh one.
    app = fake.runtime._registry.apps["stats"]
    assert app._notify == fake.notify

    await cancel_and_await(task)
    # Telegram is stopped as part of "stop intake" -- before scheduler/DB.
    assert fake.calls == ["start", "stop"]


async def test_no_telegram_interface_built_without_a_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The web dashboard always comes up; with no Telegram token configured,
    notify() stays on the Runtime WARN-log fallback."""
    _FakeTelegram.instances.clear()
    monkeypatch.setattr(serve_module, "TelegramInterface", _FakeTelegram)

    settings = make_settings(tmp_path)  # no DUDAMEL_TELEGRAM_TOKEN set
    task = asyncio.create_task(
        serve(make_orc(), settings, providers={"standard": FakeProvider([])})
    )
    await wait_for_port(settings)
    try:
        assert _FakeTelegram.instances == []
    finally:
        await cancel_and_await(task)


class _FakeTelegramThatFailsToStart(_FakeTelegram):
    """`start()` always raises -- proves a broken Telegram interface can
    never take the web dashboard down with it (spec §9: "Telegram
    unreachable -> core and dashboard unaffected")."""

    async def start(self) -> None:
        self.calls.append("start")
        raise ConnectionError("telegram is unreachable")


async def test_telegram_start_failure_does_not_prevent_web_from_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _FakeTelegram.instances.clear()
    monkeypatch.setattr(serve_module, "resolve_telegram_token", lambda settings: "fake-token")
    monkeypatch.setattr(serve_module, "TelegramInterface", _FakeTelegramThatFailsToStart)

    settings = make_settings(tmp_path)
    lockfile = settings.data_dir / ".dudamel.lock"
    task = asyncio.create_task(
        serve(make_orc(), settings, providers={"standard": FakeProvider([])})
    )
    try:
        port = await wait_for_port(settings)

        # The web surface came up and answers over the real socket despite
        # the broken Telegram interface.
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
            health = await c.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"

        assert any("telegram interface failed to start" in r.message for r in caplog.records)

        # notify() stayed on the Runtime WARN-log fallback -- bind_notify()
        # was never reached because the exception fired inside the same
        # try block, before that call.
        fake = _FakeTelegram.instances[0]
        assert fake.calls == ["start"]
        app = fake.runtime._registry.apps["stats"]
        await app._notify("still on the fallback")
        assert any("notify (no channel configured)" in r.message for r in caplog.records)
    finally:
        # Clean shutdown afterwards: the task completes without raising,
        # and shutdown must not call .stop() on a telegram that never
        # started (telegram=None on the failure path).
        await cancel_and_await(task)

    assert fake.calls == ["start"]
    assert not lockfile.exists()


# --- _InstanceLock unit coverage ----------------------------------------------


def test_instance_lock_acquire_release_roundtrip(tmp_path: Path) -> None:
    lockfile = tmp_path / ".dudamel.lock"
    lock = _InstanceLock(lockfile)
    lock.acquire()
    assert lockfile.read_text().strip() == str(os.getpid())
    lock.release()
    assert not lockfile.exists()


def test_instance_lock_exclusive_until_released(tmp_path: Path) -> None:
    """flock exclusivity (IMPORTANT finding): a second acquire() must fail
    while the first is genuinely held, and succeed again once it's
    released -- no TOCTOU window between "check" and "take" the way the old
    pid-heuristic reclaim had."""
    lockfile = tmp_path / ".dudamel.lock"
    first = _InstanceLock(lockfile)
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            _InstanceLock(lockfile).acquire()
    finally:
        first.release()

    # Released -> a fresh acquire succeeds again.
    second = _InstanceLock(lockfile)
    second.acquire()
    try:
        assert lockfile.read_text().strip() == str(os.getpid())
    finally:
        second.release()


def test_instance_lock_available_once_a_real_holder_dies(tmp_path: Path) -> None:
    """The old pid-heuristic "stale lockfile" reclaim is gone entirely
    (IMPORTANT finding): `flock` is owned by the kernel, which drops it
    unconditionally when the holding process exits, clean shutdown or
    crash alike. This spawns a real subprocess that acquires the lock,
    kills it without giving it any chance to clean up (a crash, not a
    graceful exit), and confirms a fresh acquire() in this process just
    succeeds -- no stale-pid detection step involved at all."""
    lockfile = tmp_path / ".dudamel.lock"
    script = (
        "import fcntl, os, sys, time\n"
        "fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o644)\n"
        "fcntl.flock(fd, fcntl.LOCK_EX)\n"
        "os.write(fd, str(os.getpid()).encode())\n"
        "os.fsync(fd)\n"
        "sys.stdout.write('locked\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(30)\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", script, str(lockfile)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        line = holder.stdout.readline()
        assert line.strip() == "locked"
        assert lockfile.read_text().strip() == str(holder.pid)

        holder.kill()  # SIGKILL -- no chance to unlock or clean up
        holder.wait(timeout=5.0)

        lock = _InstanceLock(lockfile)
        lock.acquire()  # must NOT raise -- the kernel already dropped the lock
        try:
            assert lockfile.read_text().strip() == str(os.getpid())
        finally:
            lock.release()
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=5.0)


# --- uvicorn internals characterization (drift alarm) --------------------------


def test_uvicorn_internals_serve_relies_on_still_exist() -> None:
    """`serve()` deliberately bypasses `uvicorn.Server.serve()` and instead
    drives `Config`/`Server` through lower-level internals directly (see
    the module docstring in `dudamel/serve.py`) -- none of which are public
    API. pyproject.toml pins `uvicorn>=0.51,<0.52` to guard against a point
    release renaming or removing any of them; this is the cheap runtime
    tripwire for that pin -- a `hasattr` check on each internal `serve()`
    actually touches, so a future upgrade attempt fails loud and fast here
    instead of surfacing as a mysterious `AttributeError` deep inside
    `serve()` at runtime.
    """
    # Methods -- present on the class regardless of instance state.
    assert hasattr(uvicorn.Config, "load")
    assert hasattr(uvicorn.Server, "startup")
    assert hasattr(uvicorn.Server, "main_loop")
    assert hasattr(uvicorn.Server, "shutdown")

    # Instance attributes -- only exist once `__init__`/`load`/prepare ran.
    config = uvicorn.Config(lambda scope, receive, send: None, port=0)
    assert hasattr(config, "loaded")
    assert config.loaded is False

    server = uvicorn.Server(config)
    assert hasattr(server, "should_exit")

    prepared = _prepare_uvicorn(config)
    assert config.loaded is True
    assert hasattr(prepared, "lifespan")


# --- forwarded-header trust: proxy_headers wiring ------------------------------


async def test_forwarded_headers_are_not_trusted_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """uvicorn enables proxy_headers by default and rewrites the client
    address from X-Forwarded-For for any peer in its trusted list -- which
    includes 127.0.0.1 out of the box. Left alone, a header would decide who
    the client is. Nothing here trusts a proxy unless one is configured."""
    captured: dict[str, object] = {}
    real_config = uvicorn.Config

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_config(*args, **kwargs)

    # Patched on the module object itself, not via the dotted string
    # "dudamel.serve.uvicorn.Config" -- pytest's string resolver walks
    # attributes starting from `dudamel`, and `dudamel.serve` is the
    # `serve` function by the time this runs (see the `serve_module`
    # comment above), not the submodule, so it can't continue on to
    # `.uvicorn.Config` from there. `uvicorn` is the same module object
    # everywhere it's imported, so patching it here reaches the
    # `uvicorn.Config(...)` call inside `serve.py` too.
    monkeypatch.setattr(uvicorn, "Config", spy)

    settings = make_settings(tmp_path)
    orc = Orchestrator(apps=[])
    task = asyncio.create_task(serve(orc, settings, providers={"standard": FakeProvider([])}))
    try:
        await wait_for_port(settings)
        assert captured["proxy_headers"] is False
        assert captured["forwarded_allow_ips"] == []
    finally:
        await cancel_and_await(task)


async def test_configured_trusted_proxies_enable_forwarded_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming a peer in `[web] trusted_proxies` is what turns forwarded-header
    trust back on, scoped to exactly the peers listed."""
    captured: dict[str, object] = {}
    real_config = uvicorn.Config

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_config(*args, **kwargs)

    monkeypatch.setattr(uvicorn, "Config", spy)  # see comment in the test above

    settings = make_settings(tmp_path)
    settings.web.trusted_proxies = ["127.0.0.1"]
    orc = Orchestrator(apps=[])
    task = asyncio.create_task(serve(orc, settings, providers={"standard": FakeProvider([])}))
    try:
        await wait_for_port(settings)
        assert captured["proxy_headers"] is True
        assert captured["forwarded_allow_ips"] == ["127.0.0.1"]
    finally:
        await cancel_and_await(task)
