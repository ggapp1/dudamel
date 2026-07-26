"""Single-process assembly (Plan 3 Task 6): the composition root that wires
`Runtime`, `JobScheduler`, the FastAPI web surface (served by uvicorn), and
the optional Telegram interface into one running process.

THIN by Global Constraints: this module owns lifecycle *sequencing* only —
zero business logic, zero LLM calls. `serve()`:

  1. Acquires an exclusive instance lock at `data_dir/.dudamel.lock` via
     `fcntl.flock` on a persistent fd — a second `serve()` against the same
     `data_dir` raises `RuntimeError` while the first is live. The kernel
     drops an `flock` unconditionally when the holding process exits, for
     any reason (clean shutdown or crash), so a leftover lockfile from a
     dead process is never mistaken for a live one and needs no pid-based
     reclaim step.
  2. Builds a `Runtime` and `await`s `start()` (DB migrations).
  3. Starts `Runtime.scheduler` — constructed but not started by `Runtime`
     itself (Task 2's design: only the assembly may start it).
  4. Builds the web app (`create_api` + `add_ui`) and binds it via uvicorn,
     in-process. `settings.web.port` is rewritten in place with the actual
     bound port once the socket exists, so a caller that requested `port=0`
     (tests) can read the real port straight off the `Settings` object it
     passed in.
  5. Builds a `TelegramInterface` and starts it IFF a bot token is
     configured (`interfaces.telegram.resolve_token`); binds
     `Runtime.bind_notify` to its `notify()` so `app.notify()` calls reach
     Telegram instead of the WARN-log fallback `Runtime` binds at
     construction. No token configured → that fallback stays in place; the
     web surface runs either way. Construction, `start()`, and the
     `bind_notify` call are wrapped in a single `try/except Exception` —
     Telegram being unreachable (bad token, network failure, anything) is
     logged and swallowed rather than taking the already-running web
     surface and scheduler down with it (spec §9: "Telegram unreachable ->
     core and dashboard unaffected"). `telegram` stays `None` on failure so
     step 6's shutdown skips it. Once this step resolves either way, an INFO
     log line reports the dashboard URL/token env var and whether Telegram
     ended up enabled — the two facts an operator needs right after startup,
     without reading source to find them.
  6. Waits for a stop signal (SIGTERM/SIGINT, or this coroutine's own task
     being cancelled), then shuts everything down in the mandated order —
     Telegram -> uvicorn -> scheduler -> Runtime (DB dispose) — releasing
     the lockfile last, in an outer `finally`, so a failure at any step
     (including during startup) can never leave a stale lock a live process
     still needs to hold.

Deliberately does NOT call `uvicorn.Server.serve()`: that method wraps its
entire body in a `capture_signals()` context manager which installs its OWN
raw `signal.signal()` SIGTERM/SIGINT handlers — unconditionally overwriting
whatever this module installs via `loop.add_signal_handler()` a few lines
above, since the uvicorn task only reaches that context manager (and thus
only clobbers ours) once this coroutine yields control, by which point our
handlers are already registered. Left alone, a SIGTERM after that point
would flip `uvicorn.Server.should_exit` directly and start closing sockets
immediately — bypassing this module's `stop_event` entirely (so the
Telegram-first shutdown ordering above is never reached) and leaving
`await stop_event.wait()` hanging forever, since nothing ever sets it.

The fix: replicate the signal-free two-thirds of `Server.serve()` that
actually matter — `Config.load()` + building `Server.lifespan` + awaiting
`Server.startup()` (binds the socket; by the time it returns
`server.servers` is already populated with the real port) — as a plain
coroutine, run `Server.main_loop()` as a background task to keep accepting
connections, and drive shutdown ourselves by flipping `should_exit` and
awaiting that task before calling `Server.shutdown()`. Same externally
visible behavior `serve()` gives a standalone process, minus the handler
hijack, so this module's own signal handling stays authoritative.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import logging
import os
import signal
from collections.abc import Awaitable
from pathlib import Path

import uvicorn

from dudamel.config import Settings
from dudamel.interfaces.telegram import TelegramInterface
from dudamel.interfaces.telegram import resolve_token as resolve_telegram_token
from dudamel.llm.provider import Provider
from dudamel.orchestrator import Orchestrator
from dudamel.runtime import Runtime
from dudamel.web.api import create_api
from dudamel.web.ui import add_ui

logger = logging.getLogger("dudamel.serve")

__all__ = ["serve"]

_LOCKFILE_NAME = ".dudamel.lock"

# Brief pause between "intake stopped" (Telegram + uvicorn no longer taking
# new work) and `scheduler.shutdown()`. APScheduler's AsyncIOExecutor.shutdown
# unconditionally cancels any still-running job future rather than honoring
# wait=True (see scheduler.py's module docstring) — once nothing can trigger
# new jobs, a short grace period lets whatever's already mid-flight finish
# naturally instead of being cut off mid-run. Best-effort and cheap by
# design, not a substitute for jobs handling their own cancellation.
_JOB_DRAIN_SECONDS = 0.1


class _InstanceLock:
    """Exclusive `serve()` lock at `path`, enforced via `fcntl.flock` on a
    persistent file descriptor.

    `flock` is owned by the open file description, and the kernel releases
    it unconditionally when the holding process exits — clean shutdown or
    crash, it doesn't matter — so there is no "the lockfile is still here
    but the pid that wrote it is dead" case left to detect and reclaim: a
    fresh `acquire()` against a dead holder's file just succeeds, the same
    way it would against a file that was never locked at all. A second
    `acquire()` while the first is genuinely live raises `BlockingIOError`
    internally, translated to the same public `RuntimeError` this always
    raised. The pid is still written into the file, but purely as an
    operator diagnostic (`cat` the lockfile to see who holds it) — nothing
    here depends on reading it back for correctness.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise RuntimeError(
                f"dudamel is already running against {self._path.parent} "
                f"(lockfile {self._path} held by a live process)"
            ) from None
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._fd = fd

    def release(self) -> None:
        if self._fd is not None:
            fd, self._fd = self._fd, None
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            with contextlib.suppress(FileNotFoundError):
                self._path.unlink()


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event
) -> list[signal.Signals]:
    """Best-effort SIGTERM/SIGINT -> `stop_event.set()`. Idempotent by
    construction: once `stop_event` is already set, a repeat signal just
    logs and returns instead of doing anything else. That matters because
    these handlers now stay installed for the ENTIRE shutdown sequence in
    `serve()`, not just up to it — a second SIGTERM landing while the first
    is still being handled (an impatient operator, or a container
    runtime's SIGTERM-then-SIGKILL grace period) must never fall through to
    default signal disposition mid-shutdown, which would hard-kill the
    process before the ordered shutdown (and the lockfile release) ever
    completes. Degrades gracefully (logs and carries on without them)
    wherever `add_signal_handler` isn't supported — e.g. off the main
    thread, or platforms without it."""

    def _handle(sig: signal.Signals) -> None:
        if stop_event.is_set():
            logger.info("shutdown already in progress; ignoring repeat %s", sig.name)
            return
        stop_event.set()

    installed = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle, sig)
        except (NotImplementedError, RuntimeError, ValueError) as e:
            logger.warning("signal handler for %s unavailable: %s", sig.name, e)
        else:
            installed.append(sig)
    return installed


def _remove_signal_handlers(loop: asyncio.AbstractEventLoop, sigs: list[signal.Signals]) -> None:
    for sig in sigs:
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            loop.remove_signal_handler(sig)


async def _stop_quietly(name: str, awaitable: Awaitable[None]) -> None:
    """Await `awaitable` as one step of the ordered shutdown; log and
    swallow any failure so it can't skip the rest of the sequence.
    `CancelledError` is a `BaseException` (not `Exception`), so it's never
    caught here — a genuine cancellation still propagates."""
    try:
        await awaitable
    except Exception as e:
        logger.warning("error stopping %s during shutdown: %s", name, e)


def _prepare_uvicorn(config: uvicorn.Config) -> uvicorn.Server:
    """Build a `Server` and replicate the signal-free setup
    `Server.serve()`/`_serve()` normally does before `startup()` — see the
    module docstring for why `serve()` is never called directly."""
    server = uvicorn.Server(config)
    if not config.loaded:
        config.load()
    server.lifespan = config.lifespan_class(config)
    return server


async def serve(
    orchestrator: Orchestrator,
    settings: Settings,
    *,
    providers: dict[str, Provider] | None = None,
) -> None:
    """Run dudamel as a single process until stopped (SIGTERM/SIGINT, or
    this coroutine's own task being cancelled — both shut down the same
    way). Raises `RuntimeError` if another `serve()` is already running
    against `settings.data_dir`.

    For `port=0` callers (tests): once the web server is bound, the real
    OS-assigned port is written back into `settings.web.port` in place, so a
    caller holding the same `settings` object can read it — after polling
    for it to become nonzero, since that happens only once this coroutine
    (typically run as a background task) actually gets to run.
    """
    lock = _InstanceLock(settings.data_dir / _LOCKFILE_NAME)
    lock.acquire()
    try:
        runtime = Runtime(orchestrator, settings, providers=providers)
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        signals_installed = _install_signal_handlers(loop, stop_event)
        server: uvicorn.Server | None = None
        web_task: asyncio.Task[None] | None = None
        telegram: TelegramInterface | None = None
        try:
            await runtime.start()
            await runtime.scheduler.start()

            app = create_api(runtime, settings)
            add_ui(app, runtime, settings)
            config = uvicorn.Config(
                app,
                host=settings.web.host,
                port=settings.web.port,
                log_level="warning",
                # Bound `server.shutdown()`'s drain wait (see the module
                # docstring's mention of the job-drain pause below) instead
                # of the uvicorn default of no timeout at all -- a
                # connection that never finishes on its own must not hang
                # the whole ordered shutdown sequence indefinitely.
                timeout_graceful_shutdown=20,
            )
            server = _prepare_uvicorn(config)
            await server.startup()
            if server.servers and server.servers[0].sockets:
                settings.web.port = server.servers[0].sockets[0].getsockname()[1]
            web_task = asyncio.create_task(server.main_loop())

            if resolve_telegram_token(settings) is not None:
                try:
                    telegram = TelegramInterface(runtime, settings)
                    await telegram.start()
                    runtime.bind_notify(telegram.notify)
                except Exception as e:
                    # Spec §9: "Telegram unreachable -> core and dashboard
                    # unaffected." A bad token, network failure, or any other
                    # construction/start blowup here must never take the web
                    # surface (already bound above) or the scheduler down
                    # with it -- log and carry on with `notify()` left at the
                    # WARN-log fallback `Runtime.__init__` already bound;
                    # `telegram` stays None so the shutdown path below skips
                    # it entirely rather than calling `.stop()` on something
                    # that never (fully) started.
                    logger.warning(
                        "telegram interface failed to start (%s); continuing without it "
                        "— web and scheduler unaffected",
                        e,
                    )
                    telegram = None

            logger.info(
                "dudamel dashboard: http://%s:%s (login with %s)",
                settings.web.host,
                settings.web.port,
                settings.web.token_env,
            )
            logger.info("telegram: %s", "enabled" if telegram is not None else "disabled")

            try:
                await stop_event.wait()
            except asyncio.CancelledError:
                pass  # task.cancel() is a stop request too — shut down clean
        finally:
            # Ordered shutdown: stop intake (Telegram, then the web server)
            # before the scheduler, then the DB last.
            # Signal handlers stay installed for this ENTIRE sequence —
            # removed only once it's fully done, in the innermost `finally`
            # below — and are idempotent (see `_install_signal_handlers`),
            # so a second SIGTERM/SIGINT arriving mid-shutdown can't restore
            # default disposition and let a third signal hard-kill the
            # process before the lockfile is released. One component
            # failing to stop cleanly must not skip the rest of the
            # sequence either — logged and swallowed, mirroring the
            # graceful-degrade rider already applied to the LLM usage-insert
            # and job_runs recording paths elsewhere in the codebase.
            try:
                if telegram is not None:
                    await _stop_quietly("telegram", telegram.stop())
                if server is not None:
                    server.should_exit = True
                    if web_task is not None:
                        await _stop_quietly("uvicorn main_loop", web_task)
                    # A failed bind means `startup()` raised before
                    # `self.servers` was ever assigned, so `shutdown()`
                    # would blow up on `for server in self.servers` with an
                    # AttributeError that has nothing to do with actual
                    # shutdown -- only call it once startup got far enough
                    # to set either flag.
                    if getattr(server, "started", False) or hasattr(server, "servers"):
                        await _stop_quietly("uvicorn shutdown", server.shutdown())
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.sleep(_JOB_DRAIN_SECONDS)
                await runtime.scheduler.shutdown()
                await _stop_quietly("runtime", runtime.stop())
            finally:
                _remove_signal_handlers(loop, signals_installed)
    finally:
        lock.release()
