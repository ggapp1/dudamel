"""Acceptance tests for dudamel/serve.py (Plan 3 Task 6): the single-process
assembly. All timing-sensitive assertions poll with a deadline -- never a
bare sleep-and-hope (Global Constraints / plan directive for this task).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from dudamel import App, Orchestrator, Runtime
from dudamel.config import Settings, TierConfig, WebConfig
from dudamel.db import Database
from dudamel.llm.testing import FakeProvider
from dudamel.models_core import JobRun
from dudamel.serve import _InstanceLock, serve

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
    """Global Constraints: web always on; Telegram absent -> notify() stays
    the Runtime WARN-log fallback."""
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


# --- _InstanceLock unit coverage ----------------------------------------------


def test_instance_lock_acquire_release_roundtrip(tmp_path: Path) -> None:
    lockfile = tmp_path / ".dudamel.lock"
    lock = _InstanceLock(lockfile)
    lock.acquire()
    assert lockfile.read_text().strip() == str(os.getpid())
    lock.release()
    assert not lockfile.exists()


def test_instance_lock_refuses_while_pid_is_alive(tmp_path: Path) -> None:
    lockfile = tmp_path / ".dudamel.lock"
    lockfile.write_text(str(os.getpid()))  # our own pid: definitely alive
    with pytest.raises(RuntimeError, match="already running"):
        _InstanceLock(lockfile).acquire()


def test_instance_lock_reclaims_a_stale_dead_pid(tmp_path: Path) -> None:
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()  # exited -- its pid is now guaranteed dead
    lockfile = tmp_path / ".dudamel.lock"
    lockfile.write_text(str(dead.pid))

    lock = _InstanceLock(lockfile)
    lock.acquire()  # must NOT raise -- the recorded pid is dead
    try:
        assert lockfile.read_text().strip() == str(os.getpid())
    finally:
        lock.release()
