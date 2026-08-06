"""APScheduler wiring: turns every registered Job into a running APScheduler
job and records EVERY outcome (ok/error/timeout/misfired) as a `job_runs`
row.

Thin wrapper — the scheduler calls only `job.fn()`. Job functions are
command-plane (unlike widgets): they may use `app.llm`/`app.notify`, but
those bindings are Runtime's concern (already wired up before this ever
fires); the scheduler itself never touches the LLM directly.

Construction only registers jobs with the underlying `AsyncIOScheduler`
(cheap, side-effect-free — safe to do in `Runtime.__init__`); nothing
actually fires until `start()`, which the single-process assembly
(dudamel.serve.serve) calls.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import UTC, datetime
from typing import Any

from apscheduler.events import EVENT_JOB_MAX_INSTANCES, EVENT_JOB_MISSED, JobExecutionEvent
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

# Upper bound on how long shutdown() waits for in-flight work to record its
# outcome. Bounded because SQLite serialises writers and db.py sets
# busy_timeout=5000 -- an unbounded drain would let one contended write stall
# the whole shutdown sequence.
_DRAIN_TIMEOUT = 2.0


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class JobScheduler:
    """Wraps an `AsyncIOScheduler`. `Runtime` owns one instance per process
    (created but not started); only the assembly calls `start()`."""

    def __init__(self, registry: Registry, db: Database) -> None:
        self._db = db
        self._scheduler = AsyncIOScheduler()
        self._scheduler.add_listener(self._on_missed, EVENT_JOB_MISSED)
        self._scheduler.add_listener(self._on_max_instances, EVENT_JOB_MAX_INSTANCES)
        # Keeps fire-and-forget misfire-recording tasks alive until they
        # finish (asyncio only holds a weak reference to a bare create_task()
        # result, so without this a task can be garbage-collected mid-flight).
        self._background_tasks: set[asyncio.Task[None]] = set()
        # Job executions currently running. shutdown() waits on these: the
        # cancelled-job row below is written from inside the cancellation
        # handler, and the assembly disposes the database as soon as
        # shutdown() returns.
        self._inflight: set[asyncio.Task[None]] = set()
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
        task = asyncio.current_task()
        if task is not None:
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)
        started = _utcnow()
        try:
            await asyncio.wait_for(job.fn(), timeout=job.timeout)
        except TimeoutError:
            detail = f"job {job.id} timed out after {job.timeout}s"
            logger.warning(detail)
            await self._record(job.id, "timeout", started, detail)
        except asyncio.CancelledError:
            # The executor cancels in-flight jobs at shutdown. Record the
            # outcome, then let the cancellation continue -- a run that was
            # killed is still a run, and the contract is that every outcome
            # lands in job_runs. The recording gets its own guard so that a
            # failure to WRITE can never replace the cancellation with a
            # different exception.
            detail = f"job {job.id} was cancelled before it finished"
            logger.info(detail)
            try:
                await self._record(job.id, "cancelled", started, detail)
            except Exception as e:  # noqa: BLE001 — recording is best-effort
                logger.warning("failed to record cancelled run for job %s: %s", job.id, e)
            raise
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

    def _on_max_instances(self, event: JobExecutionEvent) -> None:
        """APScheduler listener (sync callback, invoked from the running
        event loop by AsyncIOScheduler) for EVENT_JOB_MAX_INSTANCES — the job
        was skipped because a previous invocation is still running (exceeds
        max_instances=1), so _run_job never ran and never got a chance to
        record anything itself."""
        started = _utcnow()
        detail = "max_instances reached; run skipped"
        task = asyncio.create_task(self._record(event.job_id, "skipped", started, detail))
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

    def list_jobs(self) -> list[dict[str, Any]]:
        """Registered jobs with a best-effort next-fire time, for the
        dashboard's /jobs page. Computed from each job's trigger directly
        rather than read off the APScheduler Job's `next_run_time`
        attribute: APScheduler leaves that attribute entirely UNSET (not
        None -- absent) on a job added before `start()` runs, and since the
        scheduler is only started by the single-process assembly, the
        dashboard must still work with a constructed-but-never-started
        scheduler."""
        now = datetime.now(UTC)
        jobs = []
        for job_id, aps_job in self._aps_jobs.items():
            next_run_time = getattr(aps_job, "next_run_time", None)
            if next_run_time is None:
                next_run_time = aps_job.trigger.get_next_fire_time(None, now)
            jobs.append({"id": job_id, "next_run_time": next_run_time})
        return jobs

    async def start(self) -> None:
        self._scheduler.start()

    async def shutdown(self) -> None:
        """Stop the scheduler and wait for in-flight work to finish recording.

        `AsyncIOScheduler.shutdown()` is synchronous: it cancels each running
        job's future and returns without awaiting anything. Left there, this
        coroutine would never yield, so a cancelled job's handler -- and the
        `job_runs` row it writes -- would not run until some later suspension
        point, which the single-process assembly reaches only inside
        `Runtime.stop()`, after the database engine has been disposed. So the
        drain below is what makes the recording real rather than a write
        racing teardown.

        Bounded and best-effort: shutdown must always complete.
        """
        # A cancelled job re-raises CancelledError so the cancellation stays
        # honest; APScheduler's executor logs that at ERROR with a full
        # traceback. During a clean shutdown that is noise, not a fault --
        # suppressed for this call only, never for the process.
        aps_logger = logging.getLogger("apscheduler.executors.default")
        previous_level = aps_logger.level
        aps_logger.setLevel(logging.CRITICAL)
        try:
            if self._scheduler.running:
                self._scheduler.shutdown()
            pending = [t for t in (*self._inflight, *self._background_tasks) if not t.done()]
            if not pending:
                return
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), timeout=_DRAIN_TIMEOUT
                )
            except TimeoutError:
                logger.warning(
                    "%d job task(s) did not finish recording within %ss of shutdown",
                    len(pending),
                    _DRAIN_TIMEOUT,
                )
        finally:
            aps_logger.setLevel(previous_level)
