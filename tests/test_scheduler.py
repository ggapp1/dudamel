"""Acceptance tests for dudamel/scheduler.py::JobScheduler."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select

from dudamel import App, Orchestrator
from dudamel.config import Settings, resolve_timezone
from dudamel.db import Database
from dudamel.migrate import upgrade_core
from dudamel.models_core import JobRun
from dudamel.scheduler import JobScheduler


async def make_db(tmp_path: Path) -> Database:
    url = f"sqlite+aiosqlite:///{tmp_path}/sched.db"
    await asyncio.to_thread(upgrade_core, url)
    return Database(url)


async def poll_job_runs(
    db: Database, job_id: str, *, count: int = 1, timeout: float = 5.0
) -> list[JobRun]:
    """Poll job_runs for `job_id` until at least `count` rows exist or the
    timeout elapses -- never a single sleep-and-hope."""
    deadline = time.monotonic() + timeout
    while True:
        async with db.session() as s:
            rows = list(
                (await s.execute(select(JobRun).where(JobRun.job_id == job_id))).scalars().all()
            )
        if len(rows) >= count:
            return rows
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"expected >= {count} job_runs row(s) for {job_id!r} within {timeout}s, "
                f"got {len(rows)}"
            )
        await asyncio.sleep(0.02)


async def test_interval_job_fires_and_records_ok_row(tmp_path) -> None:
    app = App("stats", description="d")
    calls = []

    @app.job(interval_seconds=0)
    async def tick() -> None:
        calls.append(1)

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db, timezone=ZoneInfo("UTC"))
    await sched.start()
    try:
        rows = await poll_job_runs(db, "stats.tick")
    finally:
        await sched.shutdown()
        await db.dispose()

    assert rows[0].status == "ok"
    assert rows[0].finished_at is not None
    assert calls  # the job function actually ran


async def test_raising_job_records_error_row_with_traceback(tmp_path) -> None:
    app = App("stats", description="d")

    @app.job(interval_seconds=0)
    async def boom() -> None:
        raise RuntimeError("kaboom")

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db, timezone=ZoneInfo("UTC"))
    await sched.start()
    try:
        rows = await poll_job_runs(db, "stats.boom")
    finally:
        await sched.shutdown()
        await db.dispose()

    row = rows[0]
    assert row.status == "error"
    assert row.detail is not None
    assert "RuntimeError" in row.detail
    assert "kaboom" in row.detail
    assert "Traceback" in row.detail  # a real traceback snippet, not just repr(e)


async def test_slow_job_times_out(tmp_path) -> None:
    app = App("stats", description="d")

    @app.job(interval_seconds=0, timeout=0.05)
    async def slow() -> None:
        await asyncio.sleep(5)

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db, timezone=ZoneInfo("UTC"))
    await sched.start()
    try:
        rows = await poll_job_runs(db, "stats.slow")
    finally:
        await sched.shutdown()
        await db.dispose()

    row = rows[0]
    assert row.status == "timeout"
    assert row.detail is not None
    assert "0.05" in row.detail


async def test_job_that_raises_its_own_timeouterror_is_an_error_not_a_timeout(
    tmp_path,
) -> None:
    """A job that raises TimeoutError itself (e.g. an OS-level connect timeout,
    which since Python 3.10 IS TimeoutError) well within its own budget must
    be recorded as an "error" WITH the traceback -- not misreported as a
    scheduler "timeout after {job.timeout}s" the job never actually hit."""
    app = App("stats", description="d")

    @app.job(interval_seconds=0, timeout=30)
    async def flaky() -> None:
        raise TimeoutError("connection to api.example.com timed out")

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db, timezone=ZoneInfo("UTC"))
    await sched.start()
    try:
        rows = await poll_job_runs(db, "stats.flaky")
    finally:
        await sched.shutdown()
        await db.dispose()

    row = rows[0]
    assert row.status == "error"  # NOT "timeout"
    assert row.detail is not None
    assert "TimeoutError" in row.detail
    assert "api.example.com" in row.detail
    assert "Traceback" in row.detail  # the real traceback, not a fabricated line
    assert "timed out after 30" not in row.detail  # never the scheduler's message


async def test_record_swallows_any_db_error_not_just_operationalerror(tmp_path) -> None:
    """`_record` is best-effort: a DB failure while writing the job_runs row
    must never propagate into APScheduler's executor. The guard covers every
    SQLAlchemyError (e.g. InterfaceError from a straggler writing after the
    engine is disposed at shutdown), not only OperationalError."""
    from sqlalchemy.exc import SQLAlchemyError

    app = App("stats", description="d")
    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db, timezone=ZoneInfo("UTC"))

    class _BrokenDB:
        def session(self):
            raise SQLAlchemyError("engine already disposed")

    sched._db = _BrokenDB()
    try:
        # Must return normally, not raise.
        await sched._record("stats.x", "ok", datetime.now(UTC).replace(tzinfo=None), None)
    finally:
        await db.dispose()


async def test_cron_job_registers_with_correct_next_fire(tmp_path) -> None:
    app = App("stats", description="d")

    @app.job(cron="30 4 * * *")
    async def nightly() -> None:
        pass

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db, timezone=ZoneInfo("UTC"))
    try:
        aps_job = sched._aps_jobs["stats.nightly"]
        assert isinstance(aps_job.trigger, CronTrigger)

        tz = aps_job.trigger.timezone
        now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=tz)
        next_fire = aps_job.trigger.get_next_fire_time(None, now)
        assert next_fire == datetime(2026, 1, 1, 4, 30, 0, tzinfo=tz)

        # never started -- no run should ever be recorded
        async with db.session() as s:
            rows = (
                (await s.execute(select(JobRun).where(JobRun.job_id == "stats.nightly")))
                .scalars()
                .all()
            )
        assert rows == []
    finally:
        await db.dispose()


async def test_interval_job_uses_interval_trigger(tmp_path) -> None:
    app = App("stats", description="d")

    @app.job(interval_seconds=45)
    async def poll() -> None:
        pass

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db, timezone=ZoneInfo("UTC"))
    try:
        aps_job = sched._aps_jobs["stats.poll"]
        assert isinstance(aps_job.trigger, IntervalTrigger)
        assert aps_job.trigger.interval.total_seconds() == 45
    finally:
        await db.dispose()


async def test_job_registered_with_global_constraint_settings(tmp_path) -> None:
    """Every registered job must set coalesce/max_instances/misfire_grace_time
    consistently, regardless of trigger type — a mis-set value on any one of
    them lets a slow-running job pile up overlapping executions."""
    app = App("stats", description="d")

    @app.job(interval_seconds=60)
    async def poll() -> None:
        pass

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db, timezone=ZoneInfo("UTC"))
    try:
        aps_job = sched._aps_jobs["stats.poll"]
        assert aps_job.coalesce is True
        assert aps_job.max_instances == 1
        assert aps_job.misfire_grace_time == 3600
    finally:
        await db.dispose()


async def test_misfire_listener_records_misfired_row(tmp_path) -> None:
    """Directly exercises the EVENT_JOB_MISSED listener (real misfires need
    wall-clock manipulation to trigger; the listener itself is unit-tested
    here rather than waited for)."""
    from apscheduler.events import EVENT_JOB_MISSED, JobExecutionEvent

    app = App("stats", description="d")

    @app.job(interval_seconds=60)
    async def poll() -> None:
        pass

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db, timezone=ZoneInfo("UTC"))
    try:
        event = JobExecutionEvent(
            code=EVENT_JOB_MISSED,
            job_id="stats.poll",
            jobstore="default",
            scheduled_run_time=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
        )
        sched._on_missed(event)
        rows = await poll_job_runs(db, "stats.poll")
    finally:
        await db.dispose()

    assert rows[0].status == "misfired"


async def test_max_instances_listener_records_skipped_row(tmp_path) -> None:
    """Directly exercises the EVENT_JOB_MAX_INSTANCES listener (real
    max-instances collisions need concurrent execution to trigger; the
    listener itself is unit-tested here rather than waited for)."""
    from apscheduler.events import EVENT_JOB_MAX_INSTANCES, JobExecutionEvent

    app = App("stats", description="d")

    @app.job(interval_seconds=60)
    async def poll() -> None:
        pass

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db, timezone=ZoneInfo("UTC"))
    try:
        event = JobExecutionEvent(
            code=EVENT_JOB_MAX_INSTANCES,
            job_id="stats.poll",
            jobstore="default",
            scheduled_run_time=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
        )
        sched._on_max_instances(event)
        rows = await poll_job_runs(db, "stats.poll")
    finally:
        await db.dispose()

    assert rows[0].status == "skipped"


async def test_list_jobs_reports_next_fire_before_scheduler_starts(tmp_path) -> None:
    """The /jobs dashboard page needs a next-fire estimate even before the
    scheduler starts: APScheduler leaves a pending job's `next_run_time`
    attribute entirely unset until `start()` runs, so list_jobs() must fall
    back to computing it from the trigger directly."""
    app = App("stats", description="d")

    @app.job(cron="30 4 * * *")
    async def nightly() -> None:
        pass

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db, timezone=ZoneInfo("UTC"))
    try:
        jobs = sched.list_jobs()
        assert [j["id"] for j in jobs] == ["stats.nightly"]
        assert jobs[0]["next_run_time"] is not None
    finally:
        await db.dispose()


async def test_list_jobs_reflects_real_next_run_time_once_started(tmp_path) -> None:
    app = App("stats", description="d")

    @app.job(interval_seconds=3600)
    async def poll() -> None:
        pass

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db, timezone=ZoneInfo("UTC"))
    await sched.start()
    try:
        jobs = sched.list_jobs()
        assert jobs[0]["id"] == "stats.poll"
        assert jobs[0]["next_run_time"] == sched._aps_jobs["stats.poll"].next_run_time
    finally:
        await sched.shutdown()
        await db.dispose()


async def test_start_then_shutdown_is_idempotent_safe(tmp_path) -> None:
    orc = Orchestrator(apps=[])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db, timezone=ZoneInfo("UTC"))
    await sched.start()
    await sched.shutdown()
    # shutdown() again (never started a second time) must not raise
    await sched.shutdown()
    await db.dispose()


async def test_job_cancelled_at_shutdown_records_a_row(tmp_path) -> None:
    """A job still running when the scheduler shuts down is cancelled by
    APScheduler's executor. CancelledError is a BaseException, so the ordinary
    error path never sees it -- without explicit handling the run vanishes
    from job_runs entirely. The row must also be WRITTEN before shutdown()
    returns: serve() disposes the database immediately afterwards."""
    app = App("stats", description="d")
    started = asyncio.Event()

    @app.job(interval_seconds=0, timeout=30)
    async def longrun() -> None:
        started.set()
        await asyncio.sleep(30)

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db, timezone=ZoneInfo("UTC"))
    await sched.start()
    try:
        await asyncio.wait_for(started.wait(), timeout=5.0)
        await sched.shutdown()

        # No polling: the row must already exist the instant shutdown returns.
        async with db.session() as s:
            rows = (
                (await s.execute(select(JobRun).where(JobRun.job_id == "stats.longrun")))
                .scalars()
                .all()
            )
        assert [r.status for r in rows] == ["cancelled"]
        assert rows[0].finished_at is not None
    finally:
        await db.dispose()


async def test_shutdown_drains_listener_recording_tasks(tmp_path) -> None:
    """The misfire/max-instances listeners record their rows in fire-and-forget
    tasks. Those have the same hole: a misfire detected during shutdown loses
    its row unless shutdown drains them too."""
    from apscheduler.events import EVENT_JOB_MISSED, JobExecutionEvent

    app = App("stats", description="d")

    @app.job(interval_seconds=60)
    async def poll() -> None:
        pass

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db, timezone=ZoneInfo("UTC"))
    await sched.start()
    try:
        sched._on_missed(
            JobExecutionEvent(
                code=EVENT_JOB_MISSED,
                job_id="stats.poll",
                jobstore="default",
                scheduled_run_time=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
            )
        )
        await sched.shutdown()

        async with db.session() as s:
            rows = (
                (await s.execute(select(JobRun).where(JobRun.job_id == "stats.poll")))
                .scalars()
                .all()
            )
        assert [r.status for r in rows] == ["misfired"]
    finally:
        await db.dispose()


def _next_fire(scheduler: JobScheduler, job_id: str, after: datetime) -> datetime:
    """The trigger's own answer, as an absolute instant.

    Deliberately NOT an assertion about `scheduler.timezone`: apscheduler
    applies the scheduler's zone only when IT builds a trigger from a string
    alias, so that attribute reads correct while every job fires in the host's
    zone. `_aps_jobs` is the documented way to reach a trigger before start().
    """
    return scheduler._aps_jobs[job_id].trigger.get_next_fire_time(None, after)


async def test_cron_fires_in_the_configured_zone(tmp_path: Path) -> None:
    app = App("stats", description="d")

    @app.job(cron="0 20 * * *")
    async def nightly() -> None:
        return None

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db, timezone=ZoneInfo("Pacific/Auckland"))
    try:
        after = datetime(2026, 8, 27, 0, 0, tzinfo=UTC)
        assert _next_fire(sched, "stats.nightly", after) == datetime(
            2026, 8, 27, 20, 0, tzinfo=ZoneInfo("Pacific/Auckland")
        )
        assert sched._aps_jobs["stats.nightly"].trigger.timezone == ZoneInfo("Pacific/Auckland")
    finally:
        await db.dispose()


async def test_an_unset_timezone_keeps_firing_where_it_used_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Goes through `Settings()` and `resolve_timezone`, not through a zone the
    test picked. The promise is 'an existing install's jobs do not move', and
    that promise lives in the DEFAULT, so the default is what this exercises.

    The host zone is PINNED. Without that, this passes against the unfixed code
    on any UTC+12 machine -- measured -- so the red would depend on who ran it.
    """
    monkeypatch.setattr("dudamel.config.get_localzone", lambda: ZoneInfo("America/Sao_Paulo"))
    app = App("stats", description="d")

    @app.job(cron="0 20 * * *")
    async def nightly() -> None:
        return None

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    settings = Settings(database_url=f"sqlite+aiosqlite:///{tmp_path}/s.db")
    sched = JobScheduler(orc.registry, db, timezone=resolve_timezone(settings))
    try:
        after = datetime(2026, 8, 27, 0, 0, tzinfo=UTC)
        assert _next_fire(sched, "stats.nightly", after) == datetime(
            2026, 8, 27, 20, 0, tzinfo=ZoneInfo("America/Sao_Paulo")
        )
    finally:
        await db.dispose()
