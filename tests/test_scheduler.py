"""Acceptance tests for dudamel/scheduler.py::JobScheduler (Plan 3 Task 2)."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path

from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select

from dudamel import App, Orchestrator
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
    sched = JobScheduler(orc.registry, db)
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
    sched = JobScheduler(orc.registry, db)
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
    sched = JobScheduler(orc.registry, db)
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


async def test_cron_job_registers_with_correct_next_fire(tmp_path) -> None:
    app = App("stats", description="d")

    @app.job(cron="30 4 * * *")
    async def nightly() -> None:
        pass

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db)
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
    sched = JobScheduler(orc.registry, db)
    try:
        aps_job = sched._aps_jobs["stats.poll"]
        assert isinstance(aps_job.trigger, IntervalTrigger)
        assert aps_job.trigger.interval.total_seconds() == 45
    finally:
        await db.dispose()


async def test_job_registered_with_global_constraint_settings(tmp_path) -> None:
    """coalesce/max_instances/misfire_grace_time per Global Constraints."""
    app = App("stats", description="d")

    @app.job(interval_seconds=60)
    async def poll() -> None:
        pass

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db)
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
    sched = JobScheduler(orc.registry, db)
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
    sched = JobScheduler(orc.registry, db)
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
    """Plan 3 Task 4 (/jobs dashboard page): APScheduler leaves a pending
    job's `next_run_time` attribute entirely unset until `start()` runs, so
    list_jobs() must fall back to computing it from the trigger directly."""
    app = App("stats", description="d")

    @app.job(cron="30 4 * * *")
    async def nightly() -> None:
        pass

    orc = Orchestrator(apps=[app])
    db = await make_db(tmp_path)
    sched = JobScheduler(orc.registry, db)
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
    sched = JobScheduler(orc.registry, db)
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
    sched = JobScheduler(orc.registry, db)
    await sched.start()
    await sched.shutdown()
    # shutdown() again (never started a second time) must not raise
    await sched.shutdown()
    await db.dispose()
