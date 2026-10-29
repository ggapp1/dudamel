"""APScheduler wiring: turns every registered Job into a running APScheduler
job and records EVERY outcome (ok/error/timeout/skipped/misfired/cancelled)
as a `job_runs` row. A run that starts leaves a `running` row immediately,
updated in place when it ends, so a process that dies mid-run still leaves
evidence; `start()` turns what a dead process left into `interrupted` and
accounts for fires missed while nothing was running as one `misfired` row.

Thin wrapper — the scheduler calls only `job.fn()`. Job functions are
command-plane (unlike widgets, which `App.llm` refuses): they may use `app.llm`/`app.notify`, but
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
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.events import EVENT_JOB_MAX_INSTANCES, EVENT_JOB_MISSED, JobExecutionEvent
from apscheduler.job import Job as APSJob
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import func, select, update
from sqlalchemy.exc import SQLAlchemyError

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

# Counting missed fires is a loop over the trigger; an install that was off for
# a year with a per-minute job must not spin through half a million iterations
# at startup just to print a number nobody needs to the digit.
_MISSED_COUNT_CAP = 1000


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class JobScheduler:
    """Wraps an `AsyncIOScheduler`. `Runtime` owns one instance per process
    (created but not started); only the assembly calls `start()`."""

    def __init__(self, registry: Registry, db: Database, timezone: ZoneInfo) -> None:
        self._db = db
        self._timezone = timezone
        # The scheduler's own zone does not reach the triggers below (they are
        # handed in already built), but it is set anyway so a job added by any
        # other route does not silently pick up the host zone instead.
        self._scheduler = AsyncIOScheduler(timezone=timezone)
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
        # The zone goes on the TRIGGER. apscheduler applies the scheduler's own
        # zone only when it builds a trigger from a string alias; a trigger
        # handed in already built keeps whatever zone it was constructed with,
        # which for `from_crontab` with no argument is `get_localzone()`.
        trigger = (
            CronTrigger.from_crontab(job.cron, timezone=self._timezone)
            if job.cron is not None
            else IntervalTrigger(seconds=job.interval_seconds, timezone=self._timezone)
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
        # Written before the job runs, not after: a row that only appears when
        # the run finishes is exactly the row a crash, SIGKILL or power loss
        # never gets to write.
        run_id = await self._begin(job.id, started)
        try:
            # `asyncio.timeout` lets us tell OUR deadline apart from a
            # TimeoutError the job raised itself. Since Python 3.11
            # asyncio.TimeoutError IS the builtin TimeoutError (and OSError
            # ETIMEDOUT/socket.timeout are too), so a job whose own network
            # call times out raises TimeoutError from job.fn() -- catching that
            # as a scheduler timeout would record a fabricated "timed out after
            # {job.timeout}s" and drop the real traceback. `cm.expired()` is
            # true only when the context manager's own deadline fired.
            async with asyncio.timeout(job.timeout) as cm:
                await job.fn()
        except TimeoutError as e:
            if cm.expired():
                detail = f"job {job.id} timed out after {job.timeout}s"
                logger.warning(detail)
                await self._record(job.id, "timeout", started, detail, run_id=run_id)
            else:
                # The job raised its own TimeoutError -- a job error, recorded
                # with the traceback like any other raised exception.
                await self._record_error(job.id, started, e, run_id)
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
                await self._record(job.id, "cancelled", started, detail, run_id=run_id)
            except Exception as e:  # noqa: BLE001 — recording is best-effort
                logger.warning("failed to record cancelled run for job %s: %s", job.id, e)
            raise
        except Exception as e:  # job bugs must not kill the scheduler
            await self._record_error(job.id, started, e, run_id)
        else:
            await self._record(job.id, "ok", started, None, run_id=run_id)

    async def _record_error(
        self, job_id: str, started: datetime, e: BaseException, run_id: int | None = None
    ) -> None:
        """Record a job that raised as an "error" row, capturing the active
        traceback. Must be called from inside the handling `except` block so
        `traceback.format_exc()` refers to `e`."""
        detail = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"[:_DETAIL_CAP]
        logger.warning("job %s raised: %s", job_id, e)
        await self._record(job_id, "error", started, detail, run_id=run_id)

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

    async def _begin(self, job_id: str, started_at: datetime) -> int | None:
        """Insert the `running` row for a run that is starting; returns its id.

        None when the write failed -- the run still goes ahead (a job must not
        be skipped because its bookkeeping could not be written) and `_record`
        then inserts a complete row at the end instead, so the outcome is
        still recorded exactly once.
        """
        try:
            async with self._db.session() as s:
                row = JobRun(job_id=job_id, status="running", started_at=started_at)
                s.add(row)
                await s.flush()
                return row.id
        except SQLAlchemyError as e:
            logger.warning("failed to record running row for job %s: %s", job_id, e)
            return None

    async def _record(
        self,
        job_id: str,
        status: str,
        started_at: datetime,
        detail: str | None,
        *,
        run_id: int | None = None,
    ) -> None:
        """Record an outcome: finish the `running` row `run_id` names, or
        insert a complete row when there is none (skipped/misfired never
        started a run; a failed `_begin` left no row to finish)."""
        try:
            async with self._db.session() as s:
                row = await s.get(JobRun, run_id) if run_id is not None else None
                if row is None:
                    s.add(
                        JobRun(
                            job_id=job_id,
                            status=status,
                            started_at=started_at,
                            finished_at=_utcnow(),
                            detail=detail,
                        )
                    )
                else:
                    row.status = status
                    row.finished_at = _utcnow()
                    row.detail = detail
        except SQLAlchemyError as e:
            # The job itself already ran to whatever conclusion it reached; a
            # DB failure recording that outcome must not raise into the
            # scheduler's executor. Broad by design (any SQLAlchemy error, not
            # just OperationalError): the ok/error/timeout paths call _record
            # directly, and an InterfaceError from a straggler task writing
            # after the engine is disposed at shutdown, or an IntegrityError,
            # would otherwise surface as a spurious executor failure -- the
            # same reason the cancelled path already catches broadly.
            logger.warning("failed to record job_runs row for job %s: %s", job_id, e)

    async def _reconcile_history(self) -> None:
        """Make `job_runs` honest about the time the process was not running.
        Called by `start()` before anything is scheduled. Best-effort: a
        database failure here is logged and must not stop the scheduler."""
        try:
            await self._mark_interrupted()
            await self._record_downtime_misfires()
        except SQLAlchemyError as e:
            logger.warning("failed to reconcile job_runs at startup: %s", e)

    async def _mark_interrupted(self) -> None:
        """A row still `running` at startup belongs to a process that died
        mid-run. Nothing else can own it: `serve()` holds the exclusive
        `data_dir/.dudamel.lock` for the life of the process, so no second
        instance is executing jobs against this database, and this scheduler
        has not started anything yet."""
        async with self._db.session() as s:
            result = await s.execute(
                update(JobRun)
                .where(JobRun.status == "running")
                .values(
                    status="interrupted",
                    finished_at=_utcnow(),
                    detail="the process ended while this run was in progress "
                    "(crash, kill or power loss); the outcome is unknown",
                )
            )
        if result.rowcount:
            logger.warning(
                "%d job run(s) were interrupted by the previous shutdown", result.rowcount
            )

    async def _record_downtime_misfires(self) -> None:
        """One `misfired` row per job that should have fired while the process
        was down, saying how many fires were missed.

        Needed because the in-memory job store recomputes every job's next fire
        from "now" at each start: APScheduler never learns a fire was due while
        we were away, so its own EVENT_JOB_MISSED (which `_on_missed` records)
        cannot fire for downtime. That listener covers the other case -- a fire
        that came due while the process WAS up but ran more than
        `misfire_grace_time` late. The two windows do not overlap: this one
        ends at the moment of startup, before the scheduler can run anything.

        The window starts at the job's latest recorded run (any status, so a
        previous startup's `misfired` row moves it forward and the same
        downtime is never reported twice). A job with no rows is new or the
        database is fresh: nothing was missed, nothing is recorded.
        """
        now = _utcnow()
        async with self._db.session() as s:
            last_by_job = dict(
                (
                    await s.execute(
                        select(JobRun.job_id, func.max(JobRun.started_at)).group_by(JobRun.job_id)
                    )
                ).all()
            )
        for job_id, aps_job in self._aps_jobs.items():
            last = last_by_job.get(job_id)
            if last is None:
                continue
            missed = self._count_missed(aps_job, last.replace(tzinfo=UTC), now.replace(tzinfo=UTC))
            if missed == 0:
                continue
            count = f"{_MISSED_COUNT_CAP}+" if missed > _MISSED_COUNT_CAP else str(missed)
            await self._record(
                job_id,
                "misfired",
                now,
                f"{count} scheduled run(s) were missed while dudamel was not running "
                f"(last run started {last:%Y-%m-%d %H:%M:%S} UTC)",
            )

    @staticmethod
    def _count_missed(aps_job: APSJob, last: datetime, now: datetime) -> int:
        """Fires of this job's trigger in (last, now], capped at one past
        `_MISSED_COUNT_CAP`. Uses the trigger the job was registered with, so
        the framework zone it carries decides what "every day at 20:00" means."""
        trigger = aps_job.trigger
        if isinstance(trigger, IntervalTrigger):
            # An interval trigger is anchored at its own construction time --
            # which is this startup -- so it cannot say when it would have
            # fired last week. Intervals count from the last run instead.
            step = trigger.interval.total_seconds()
            if step <= 0:
                return 0
            return min(int((now - last).total_seconds() // step), _MISSED_COUNT_CAP + 1)
        count = 0
        fire = trigger.get_next_fire_time(None, last + timedelta(microseconds=1))
        while fire is not None and fire <= now and count <= _MISSED_COUNT_CAP:
            count += 1
            fire = trigger.get_next_fire_time(fire, fire + timedelta(microseconds=1))
        return count

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
        # Before the scheduler can run anything, so nothing it starts is
        # mistaken for a leftover `running` row.
        await self._reconcile_history()
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
