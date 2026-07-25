"""APScheduler wiring (Plan 3 Task 2): turns every registered Job into a
running APScheduler job and records EVERY outcome (ok/error/timeout/misfired)
as a `job_runs` row.

Thin wrapper — the scheduler calls only `job.fn()`. Job functions are
command-plane (unlike widgets): they may use `app.llm`/`app.notify`, but
those bindings are Runtime's concern (already wired up before this ever
fires); the scheduler itself never touches the LLM directly.

Construction only registers jobs with the underlying `AsyncIOScheduler`
(cheap, side-effect-free — safe to do in `Runtime.__init__`); nothing
actually fires until `start()`, which the assembly (Plan 3 Task 6) calls.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import UTC, datetime

from apscheduler.events import EVENT_JOB_MISSED, JobExecutionEvent
from apscheduler.job import Job as APSJob
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.exc import OperationalError

from dudamel.contract.types import Job
from dudamel.db import Database
from dudamel.models_core import JobRun
from dudamel.registry import Registry

logger = logging.getLogger("dudamel.scheduler")

# job_runs.detail is unbounded Text, but a pathological traceback (deep
# recursion, huge repr in a local) shouldn't be allowed to bloat one row.
_DETAIL_CAP = 4000


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class JobScheduler:
    """Wraps an `AsyncIOScheduler`. `Runtime` owns one instance per process
    (created but not started); only the assembly calls `start()`."""

    def __init__(self, registry: Registry, db: Database) -> None:
        self._db = db
        self._scheduler = AsyncIOScheduler()
        self._scheduler.add_listener(self._on_missed, EVENT_JOB_MISSED)
        # Keeps fire-and-forget misfire-recording tasks alive until they
        # finish (asyncio only holds a weak reference to a bare create_task()
        # result, so without this a task can be garbage-collected mid-flight).
        self._background_tasks: set[asyncio.Task[None]] = set()
        # Exposed (privately) for introspection: APScheduler jobs added
        # before the scheduler is started are merely "pending" and are not
        # reachable via scheduler.get_job(), so callers/tests that need the
        # trigger before start() (e.g. to assert next-fire time) go through
        # this dict instead.
        self._aps_jobs: dict[str, APSJob] = {job.id: self._register(job) for job in registry.jobs}

    def _register(self, job: Job) -> APSJob:
        trigger = (
            CronTrigger.from_crontab(job.cron)
            if job.cron is not None
            else IntervalTrigger(seconds=job.interval_seconds)
        )
        return self._scheduler.add_job(
            self._run_job,
            trigger=trigger,
            id=job.id,
            args=[job],
            coalesce=True,
            max_instances=1,
            misfire_grace_time=3600,
        )

    async def _run_job(self, job: Job) -> None:
        started = _utcnow()
        try:
            await asyncio.wait_for(job.fn(), timeout=job.timeout)
        except TimeoutError:
            detail = f"job {job.id} timed out after {job.timeout}s"
            logger.warning(detail)
            await self._record(job.id, "timeout", started, detail)
        except Exception as e:  # job bugs must not kill the scheduler
            detail = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"[:_DETAIL_CAP]
            logger.warning("job %s raised: %s", job.id, e)
            await self._record(job.id, "error", started, detail)
        else:
            await self._record(job.id, "ok", started, None)

    def _on_missed(self, event: JobExecutionEvent) -> None:
        """APScheduler listener (sync callback, invoked from the running
        event loop by AsyncIOScheduler) for EVENT_JOB_MISSED — the job was
        skipped entirely (misfire_grace_time exceeded), so _run_job never ran
        and never got a chance to record anything itself."""
        started = _utcnow()
        detail = f"scheduled run at {event.scheduled_run_time} was missed"
        task = asyncio.create_task(self._record(event.job_id, "misfired", started, detail))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _record(
        self, job_id: str, status: str, started_at: datetime, detail: str | None
    ) -> None:
        try:
            async with self._db.session() as s:
                s.add(
                    JobRun(
                        job_id=job_id,
                        status=status,
                        started_at=started_at,
                        finished_at=_utcnow(),
                        detail=detail,
                    )
                )
        except OperationalError as e:
            # The job itself already ran to whatever conclusion it reached; a
            # DB hiccup recording that outcome must not raise into the
            # scheduler's executor (mirrors the llm_calls usage-insert rider).
            logger.warning("failed to record job_runs row for job %s: %s", job_id, e)

    async def start(self) -> None:
        self._scheduler.start()

    async def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown()
