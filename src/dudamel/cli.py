"""`dudamel` command-line entry point — the product's front door: `dudamel
new/run/db migrate/doctor/apps list/token rotate`.

argparse only (no click/typer dependency). Every command below operates on
the project in the CURRENT working directory (`new` is the one exception —
it creates a project elsewhere) — this mirrors the scaffold's own README,
which always tells you to `cd` into the project first.

Errors are actionable sentences, never tracebacks: `main()` catches
`DudamelError` (and its CLI-local subclass `CliError`) and any other
exception, printing a one-line message and returning a nonzero exit code —
unless `--debug` was passed, in which case the exception is left to
propagate as a normal Python traceback.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from functools import partial
from importlib.resources import files
from pathlib import Path
from types import ModuleType
from typing import Literal

import httpx
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from dudamel import apps as suite
from dudamel._version import __version__
from dudamel.apps import missing_requirements
from dudamel.config import Settings, TierConfig, resolve_timezone, timezone_source
from dudamel.contract.renderers import ACTION_LABEL_MAX
from dudamel.exceptions import DudamelError
from dudamel.interfaces.telegram import resolve_token as resolve_telegram_token
from dudamel.llm.probe import probe_tool_calling
from dudamel.migrate import (
    _sqlite_path,
    current_heads,
    ensure_app_migrations,
    generate_app_migration,
    pending_migrations,
    project_lane_pending,
    script_heads,
    suite_lane_pending,
    sync_url,
    upgrade_database,
)
from dudamel.orchestrator import Orchestrator

# `_is_enabled` below is the resolver's own "presence means enabled" rule,
# reused rather than restated: a second copy that drifted would make `apps
# list` describe a configuration different from the one that actually runs.
from dudamel.resolve import Resolution, resolve_apps
from dudamel.resolve import _is_enabled as _suite_app_enabled
from dudamel.runtime import build_provider
from dudamel.serve import serve
from dudamel.web.api import is_loopback_host, resolve_cookie_secure
from dudamel.web.auth import resolve_token as resolve_web_token

__all__ = ["main"]

_DEFAULT_MODULE = "assistant"

_INVALID_PROJECT_NAME_CHARS = re.compile(r"[^a-z0-9._-]+")


def _sanitize_project_name(raw: str) -> str:
    """Turn a project directory name into a valid packaging project name
    (PEP 503 normalization rules: lowercase, only `[a-z0-9._-]`, no
    leading/trailing separator) for the scaffolded `pyproject.toml`'s
    `[project] name` — an arbitrary directory name (spaces, `_`, capitals,
    a leading dot) would otherwise produce a `pyproject.toml` `uv`/`pip`
    reject outright."""
    sanitized = _INVALID_PROJECT_NAME_CHARS.sub("-", raw.strip().lower()).strip("-._")
    return sanitized or "dudamel-project"


class CliError(DudamelError):
    """A user-facing CLI problem — `main()` prints it as a plain sentence."""


# --- project module discovery -----------------------------------------------
#
# `run`/`db migrate`/`doctor` all import the project's entry-point module the
# same way: add the project dir to `sys.path`, import `<module_name>.py` by
# name, and read its module-level `orchestrator`. Tracked here so a SECOND
# `dudamel` invocation against a DIFFERENT project dir in the same process
# (only possible in-process, e.g. tests exercising several scaffolds back to
# back — a real CLI invocation is always a fresh interpreter) can't silently
# resolve `assistant`/`apps` from Python's module cache to the WRONG
# project's stale modules.
_inserted_project_paths: set[str] = set()


def _import_project_module(project_dir: Path, module_name: str) -> ModuleType:
    module_path = project_dir / f"{module_name}.py"
    if not module_path.exists():
        raise CliError(
            f"{module_path} not found — is {project_dir} a dudamel project? "
            "(run `dudamel new NAME` to scaffold one)"
        )
    path_str = str(project_dir.resolve())
    for stale in _inserted_project_paths - {path_str}:
        with contextlib.suppress(ValueError):
            sys.path.remove(stale)
    _inserted_project_paths.clear()
    for name in list(sys.modules):
        if name == module_name or name.split(".", 1)[0] == "apps":
            del sys.modules[name]
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    _inserted_project_paths.add(path_str)
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise CliError(f"could not import {module_name!r} from {project_dir}: {e}") from e


def _load_orchestrator(project_dir: Path, module_name: str) -> Orchestrator:
    module = _import_project_module(project_dir, module_name)
    orchestrator = getattr(module, "orchestrator", None)
    if orchestrator is None:
        raise CliError(
            f"{module_name}.py does not define `orchestrator` — expected a "
            "module-level `orchestrator = Orchestrator(apps=[...])` (see the "
            "scaffold's assistant.py)"
        )
    if not isinstance(orchestrator, Orchestrator):
        raise CliError(
            f"{module_name}.orchestrator is a {type(orchestrator).__name__}, not an Orchestrator"
        )
    return orchestrator


def _load_dotenv_into_environ(project_dir: Path) -> None:
    """`.env` values become real process environment variables (never
    overriding one already set — same "env beats .env" precedence
    `Settings.load` uses). Required independently of `Settings.load`'s own
    dotenv handling: web/telegram token resolution and tier API-key lookup
    all read `os.environ` directly (`web/auth.py::resolve_token`,
    `interfaces/telegram.py::resolve_token`, `Runtime._build_tiers`), never
    through a `Settings` field."""
    env_path = project_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


# --- new ---------------------------------------------------------------------

_DEPLOY_TEMPLATE_NAMES = ("dudamel.plist", "dudamel.service")


def _write_deploy_templates(target: Path, project_name: str) -> None:
    """Copy the launchd/systemd supervisor templates into `<project>/deploy/`
    with `{{PROJECT_DIR}}`/`{{PROJECT_NAME}}` filled in -- both need the
    project's own absolute path (a service manager doesn't run with the cwd
    a user typed `dudamel new` from), which `scaffold_template/`'s plain
    `shutil.copytree` in `cmd_new` above has no way to supply.

    `{{UV_PATH}}` is the `uv` on PATH when the project is scaffolded: a systemd
    --user manager's PATH lacks ~/.local/bin, so the unit cannot say bare `uv`.
    Without one on PATH it falls back to systemd's `%h/.local/bin/uv`, uv's
    usual install location; the unit's own header tells the operator to check
    it on the host that will actually run it."""
    deploy_dir = target / "deploy"
    deploy_dir.mkdir(exist_ok=True)
    template_dir = Path(str(files("dudamel") / "deploy_templates"))
    uv_path = shutil.which("uv") or "%h/.local/bin/uv"
    for name in _DEPLOY_TEMPLATE_NAMES:
        text = (template_dir / name).read_text()
        text = text.replace("{{PROJECT_DIR}}", str(target.resolve()))
        text = text.replace("{{PROJECT_NAME}}", project_name)
        text = text.replace("{{UV_PATH}}", uv_path)
        (deploy_dir / name).write_text(text)


def cmd_new(args: argparse.Namespace) -> int:
    target = Path(args.name)
    if target.exists():
        if not target.is_dir():
            raise CliError(f"{target} already exists and is not a directory")
        if any(target.iterdir()):
            raise CliError(
                f"{target} already exists and is not empty; "
                "choose a different name or empty it first"
            )
    else:
        target.mkdir(parents=True)

    template_dir = Path(str(files("dudamel") / "scaffold_template"))
    shutil.copytree(template_dir, target, dirs_exist_ok=True)
    readme = target / "README.md"
    readme.write_text(readme.read_text().replace("{{PROJECT_NAME}}", args.name))

    project_name = _sanitize_project_name(target.name)
    pyproject_path = target / "pyproject.toml"
    pyproject_path.write_text(
        pyproject_path.read_text()
        .replace("{{PROJECT_NAME}}", project_name)
        .replace("{{DUDAMEL_VERSION}}", __version__)
    )

    _write_deploy_templates(target, project_name)

    env_path = target / ".env"
    env_path.write_text(f"DUDAMEL_WEB_TOKEN={secrets.token_urlsafe(32)}\n")
    os.chmod(env_path, 0o600)

    # Ships the migrations/ scaffolding (env.py + script.py.mako + an empty
    # versions/) but deliberately no pre-generated revision -- see the
    # scaffold README's Quickstart. `versions/` is committed empty via
    # .gitkeep since git doesn't track empty directories.
    ensure_app_migrations(target)
    (target / "migrations" / "versions" / ".gitkeep").touch()

    print(f"created {target}/")
    print("next steps:")
    print(f"  cd {args.name}")
    # Not `db migrate`: a fresh project registers no apps, so it has no models
    # and that command would only print `no changes`. `apps list` is what a new
    # project needs -- it shows the first-party apps that can be switched on in
    # dudamel.toml.
    print("  uv run dudamel apps list")
    print("  uv run dudamel run")
    return 0


# --- run -----------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


def cmd_run(args: argparse.Namespace) -> int:
    # A real `dudamel run` invocation is a fresh interpreter with nothing
    # else configuring logging, so INFO-and-up (DEBUG with --debug) reaching
    # the terminal by default is what makes serve()'s own startup/shutdown
    # logging (dashboard URL, telegram status, ordered-shutdown messages)
    # visible at all rather than silently discarded by logging's default
    # "no handler configured" behavior.
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATEFMT,
    )
    project_dir = Path.cwd()
    _load_dotenv_into_environ(project_dir)
    settings = Settings.load(project_dir)
    orchestrator = _load_orchestrator(project_dir, args.module)
    # strict: a misconfigured [apps.*] block must stop a real run rather than
    # start an assistant that quietly lacks the app the operator asked for.
    resolution = resolve_apps(orchestrator, settings, strict=True)
    asyncio.run(
        serve(
            # Everything that resolved -- suite apps plus the project's own.
            # `mcp` is carried over because it belongs to the project's
            # orchestrator, not to any app.
            Orchestrator(apps=resolution.apps, mcp=orchestrator.mcp),
            settings,
            suite_lanes=resolution.suite_lanes,
        )
    )
    return 0


# --- db migrate ------------------------------------------------------------


def _lane_label(pending: str) -> str:
    """A `migrate.pending_migrations` entry as a short lane name: `core`,
    `app 'tasks'` (a suite app's lane) or `project` (the project's own)."""
    label = pending.removesuffix(" schema is behind head")
    return "project" if label == "app" else label


def cmd_db_migrate(args: argparse.Namespace) -> int:
    cwd = Path.cwd()
    settings = Settings.load(cwd)
    orchestrator = _load_orchestrator(cwd, _DEFAULT_MODULE)
    resolution = resolve_apps(orchestrator, settings, strict=True)
    # The lane lives under `settings.project_dir` -- the cwd unless dudamel.toml
    # says otherwise. `assistant.py` is still imported from the cwd (that is
    # where the operator ran the command), but the migration lane has to be the
    # one the startup gate and `doctor` read, or this command writes revisions
    # into a directory nothing else ever looks at and its own remedy line
    # ("run `dudamel db migrate -m init`") can never be satisfied.
    project_dir = settings.project_dir
    # Autogenerate diffs the models against the live database, so the database
    # has to already be at head for the project's lane (and every enabled suite
    # lane) before we ask for a new revision. A fresh clone is the ordinary case
    # where it is not: revisions are committed, the db file is gitignored. Diffing
    # against that empty db re-emits CREATE TABLE as a fresh revision that then
    # collides with the committed one -- and the orphan file stays on disk, so
    # every later `db migrate` fails the same way. Safe with no `migrations/`
    # yet: `upgrade_all` skips the project lane when there is none. Core goes up
    # first, in the same call, so the schema is ready for autogenerate. One
    # backup covers this whole pass, and only if something is pending.
    # Read BEFORE the upgrade so the command can say what it applied: a first
    # `db migrate -m init` creates every table of the core schema and each
    # enabled suite app, then reports `no changes` because it generated no
    # revision of its own -- true, and baffling next to 16 new tables.
    already_pending = pending_migrations(settings.database_url, project_dir, resolution.suite_lanes)
    backup = upgrade_database(settings.database_url, project_dir, resolution.suite_lanes)
    if already_pending:
        # stderr, so stdout stays exactly what scripts parse: the new
        # revision's path, or `no changes`.
        print(f"applied: {', '.join(_lane_label(p) for p in already_pending)}", file=sys.stderr)
    # LOCAL apps only. A suite app's revisions ship in the wheel, so its tables
    # must never enter the project's own autogenerate diff: otherwise every
    # user mints a private revision for shipped code, and the shipped lane's
    # CREATE TABLE later collides with the table their own lane already made.
    path = generate_app_migration(
        Orchestrator(apps=resolution.local_apps),
        settings.database_url,
        args.message,
        project_dir,
        allow_destructive=args.allow_destructive,
    )
    # Second pass, now that the revision above exists: always bring the project
    # db to head, whether or not this call produced a new revision -- the one
    # just written must take effect without a second `db migrate`. Each enabled
    # suite app's own lane goes up with it (a no-op if already at head). If the
    # first pass already took a backup, that is the run's pre-migration snapshot
    # and the generated revision rides on it; otherwise this pass backs up
    # (only if the new revision is actually pending), so a run makes at most one.
    upgrade_database(
        settings.database_url,
        project_dir,
        resolution.suite_lanes,
        backup=backup is None,
    )
    print(str(path) if path is not None else "no changes")
    return 0


# --- doctor ------------------------------------------------------------------


# A doctor line's severity. `True` is a pass and `False` is a FAILURE: the one
# state that makes `doctor` exit 1, so it is reserved for what would stop
# `dudamel run` from starting or leave a configured feature broken. "warn" is a
# real problem that does not stop anything (a typo'd layout id is skipped at
# render time; an LLM backend that is merely not up yet) and "info" is a normal
# state that only wants a sentence (Telegram not configured, no Tailscale, a
# database `run` has yet to create). Neither moves the exit code -- otherwise
# every fresh checkout would exit non-zero and CI could not tell "broken" from
# "not set up yet".
Status = bool | Literal["warn", "info"]

_GLYPHS = {True: "✓", False: "✗", "warn": "⚠", "info": "ℹ"}


def _line(status: Status, label: str, detail: str) -> str:
    return f"{_GLYPHS[status]} {label}: {detail}"


def _sqlite_file_path(db_url: str) -> Path | None:
    """The on-disk file a SQLite URL names, or None for a non-SQLite or
    in-memory URL. Reuses `migrate._sqlite_path` (which owns the URL-parsing
    rules and raises for in-memory) so doctor's existence checks and the
    backup path can never disagree about what file a URL names."""
    try:
        return _sqlite_path(db_url)
    except DudamelError:
        return None  # :memory: / path-less sqlite -- nothing on disk to check


def _check_db_connect(db_url: str) -> tuple[Status, str]:
    # For SQLite, check if the database file exists before trying to connect.
    # Informational, not a failure: `dudamel run` creates the file, so its
    # absence on a fresh project is the normal state, not a fault.
    path = _sqlite_file_path(db_url)
    if path is not None and not path.exists():
        return "info", "not created yet (run `dudamel run` first)"
    try:
        engine = create_engine(sync_url(db_url))
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        finally:
            engine.dispose()
    except Exception as e:
        return False, f"cannot connect to {db_url!r} ({e})"
    return True, f"connected to {db_url!r}"


def _behind_status(auto_migrate: bool) -> Status:
    """How serious an unapplied migration is, judged the way startup judges it.

    With `auto_migrate` (the default) `dudamel run` applies whatever is pending
    itself, so a schema that is behind is just the next run's work: info. With
    it off the startup gate REFUSES to start on a pending tier -- that is the
    failure doctor exists to catch before a restart does."""
    return "info" if auto_migrate else False


def _check_core_migrations(db_url: str, *, auto_migrate: bool = True) -> tuple[Status, str]:
    behind = _behind_status(auto_migrate)
    # For SQLite, check if database file exists before trying to connect
    # (avoid creating an empty database file when just running doctor)
    path = _sqlite_file_path(db_url)
    if path is not None and not path.exists():
        return (
            behind,
            "not yet applied — run `dudamel run` or `dudamel db migrate -m <msg>` once",
        )
    try:
        heads = script_heads(str(files("dudamel") / "migrations"))
        current = current_heads(db_url, "alembic_version_core")
    except Exception as e:
        return False, f"could not read core migration state ({e})"
    if not heads:
        return False, "no core migration scripts found (packaging problem)"
    if current == heads:
        return True, "at head"
    if not current:
        return behind, "not yet applied — run `dudamel run` or `dudamel db migrate -m <msg>` once"
    return behind, f"behind head ({sorted(current)} != {sorted(heads)})"


def _check_app_migrations_dir(project_dir: Path, *, any_local_apps: bool) -> tuple[Status, str]:
    mig_dir = project_dir / "migrations"
    if not mig_dir.exists():
        # A failure only when a local app has models that need a lane; with
        # none there is nothing the directory would hold.
        status: Status = False if any_local_apps else "info"
        return status, "migrations/ not found — run `dudamel db migrate -m init` to create it"
    revisions = list((mig_dir / "versions").glob("*.py")) if (mig_dir / "versions").exists() else []
    if not revisions:
        # LOCAL apps, not every resolved app: `cmd_db_migrate` autogenerates
        # against `resolution.local_apps` alone, because a suite app's
        # revisions ship in the wheel and must never enter the project's own
        # diff. So a project running only enabled suite apps has models and
        # still has nothing to generate -- advising `db migrate` there is the
        # same dead end as advising it in an empty project.
        if not any_local_apps:
            return True, "present, no revisions yet — normal until an app defines models"
        return True, "present, no revisions yet — run `dudamel db migrate -m init`"
    return True, f"present ({len(revisions)} revision{'s' if len(revisions) != 1 else ''})"


def _check_pending_migrations(
    db_url: str,
    project_dir: Path,
    suite_lanes: Sequence[tuple[str, Path]],
    *,
    auto_migrate: bool = True,
) -> tuple[Status, str]:
    """Every tier the startup gate would refuse to start on.

    Deliberately the SAME call the gate makes (`migrate.pending_migrations`
    with the resolved suite lanes), so doctor cannot report a green schema on
    a project that then refuses to start -- which is exactly what an enabled
    suite app whose shipped lane is unapplied would otherwise do.
    """
    path = _sqlite_file_path(db_url)
    if path is not None and not path.exists():
        # Connecting would CREATE the file; doctor must not be what creates a
        # project's database (`_check_db_connect` already reported this).
        return (
            _behind_status(auto_migrate),
            "not created yet — run `dudamel run` or `dudamel db migrate -m <msg>` once",
        )
    try:
        pending = pending_migrations(db_url, project_dir, suite_lanes)
    except Exception as e:
        return False, f"could not read migration state ({e})"
    if pending:
        detail = "; ".join(pending)
        if auto_migrate:
            return "info", detail + " — `dudamel run` will apply them (auto_migrate is on)"
        return (
            False,
            detail + " — auto_migrate is off, so `dudamel run` would refuse to start; "
            "run `dudamel db migrate -m <msg>`",
        )
    return True, "none — every lane is at head"


def _check_tier(cfg: TierConfig) -> tuple[Status, str]:
    # Misconfiguration (no base_url, key env unset) is a failure -- the Runtime
    # refuses to build the tier. A backend that does not answer is only a
    # warning: `run` starts without it, and the usual reason is that the local
    # server (Ollama) has not been started yet.
    if cfg.provider == "openai-compatible":
        if not cfg.base_url:
            return False, "no base_url configured"
        url = cfg.base_url.rstrip("/") + "/models"
        try:
            resp = httpx.get(url, timeout=2.0)
        except httpx.HTTPError as e:
            return "warn", f"{url} unreachable ({e.__class__.__name__})"
        if resp.status_code < 400:
            return True, f"{url} reachable"
        return "warn", f"{url} returned HTTP {resp.status_code}"
    if cfg.provider == "anthropic":
        env = cfg.api_key_env or "ANTHROPIC_API_KEY"
        if os.environ.get(env):
            return True, f"{env} is set"
        return False, f"{env} is not set"
    return True, "fake provider (tests only)"


def _probe_tier_tool_calling(name: str, cfg: TierConfig) -> tuple[bool, str]:
    """Build a live Provider for tier `name` and run the tool-calling probe
    against it. Never raises: a probe that could crash `doctor` would be
    worse than no probe -- construction failures (missing base_url, unset
    API key env var) and runtime failures (unreachable backend, malformed
    reply) both degrade to a ✗ line."""
    try:
        provider = build_provider(name, cfg)
        return asyncio.run(probe_tool_calling(provider, model=cfg.model))
    except Exception as e:
        return False, f"probe could not run ({type(e).__name__}: {e})"


def _check_telegram(settings: Settings) -> tuple[Status, str]:
    token = resolve_telegram_token(settings)
    if not token:
        return "info", f"{settings.telegram.token_env} not set — disabled (optional)"
    n = len(settings.telegram.allowed_user_ids)
    groups = "allowed" if settings.telegram.allow_groups else "blocked"
    return True, f"token configured, {n} allowed user id(s), groups {groups}"


def _check_web_token(settings: Settings) -> tuple[Status, str]:
    token = resolve_web_token(settings)
    if not token:
        # Judged the way `create_api` judges it: off-box without a token it
        # REFUSES to start (a failure); on loopback it starts and only logs
        # that the dashboard cannot be logged into (a warning).
        status: Status = "warn" if is_loopback_host(settings.web.host) else False
        return status, f"{settings.web.token_env} not set — dashboard login impossible"
    return True, f"{settings.web.token_env} is set"


def _check_timezone(settings: Settings) -> tuple[bool, str]:
    """Report the RESOLVED framework zone and where it came from.

    Always OK -- an unusable zone is rejected by validation long before this
    runs. It is printed because unset means the host's zone, so the answer
    depends on /etc/localtime, and because a misspelled top-level key is dropped
    in silence: this line is where an operator sees that `timezome` did nothing.
    """
    zone = resolve_timezone(settings)
    if timezone_source(settings) == "config":
        return True, f"{zone} (from `timezone` in dudamel.toml)"
    return True, f"{zone} (from the host; set a top-level `timezone` to pin it)"


# The suite apps that key a stored row to a calendar day, in the order the
# report names them. `notes` is absent deliberately: it stores timestamps, so
# moving the zone changes how a note is displayed but never what a row means.
_DAY_KEYED_APPS = ("tasks", "habits")


def _check_day_boundary(settings: Settings, resolution: Resolution) -> tuple[Status, str] | None:
    """Report an unpinned, host-derived day boundary under the apps that store
    days -- what is true and checkable from config, and nothing more.

    Reported, not warned: nothing here is broken. The condition is a posture,
    the same shape as `_check_cookie_secure`'s derived-value lines -- these
    apps cut days somewhere the operator never chose, so the answer follows
    /etc/localtime and moves with the machine. It is worth a line because
    `tasks` and `habits` used to carry their own `timezone`, defaulting to UTC:
    on an install old enough to have run that, the days already stored were cut
    at UTC and cannot be re-cut, because a `day` column is a date and a date has
    lost the instant it came from.

    So the remedy leads with pinning the zone just resolved, which keeps every
    boundary exactly where it is today and only stops it drifting; UTC comes
    second, as the way back to the old boundary, and carries the warning that it
    moves the scheduler too.

    Gated on the apps actually being enabled, because a project running neither
    has no row keyed to any boundary -- and an unclearable line on a fresh
    install is how an operator learns to skip this report. Judged on
    `resolution.apps`, the set that will really run: a local app may not take a
    suite app's name, so a name here is that suite app.

    Config-only, with no database query: `doctor` runs before migrations and
    against projects with no database at all.

    Returns None when there is nothing to say -- no such app is on, the zone is
    pinned (the boundary is then the operator's own answer, and fixed against
    the host), or the host is UTC and matches what these apps used to do.
    """
    running = [app.name for app in resolution.apps]
    named = [name for name in _DAY_KEYED_APPS if name in running]
    if not named:
        return None
    if timezone_source(settings) == "config":
        return None
    zone = resolve_timezone(settings)
    if str(zone) == "UTC":
        return None
    return "info", (
        f"{' and '.join(named)} cut days at {zone}, the host's zone \u2014 unpinned, "
        "so it follows /etc/localtime. Set a top-level "
        f'timezone = "{zone}" in dudamel.toml to hold it; see "Upgrading" in the '
        "README if these apps have rows from a version that cut days at UTC"
    )


def _check_cookie_secure(settings: Settings) -> tuple[Status, str]:
    """Report the RESOLVED session-cookie `Secure` posture against the scheme
    doctor actually prints (always `http://` — see `_check_tailscale`).

    `[web] cookie_secure` describes the transport the browser sees, but its
    auto-derivation can only read the bind host, so the derived value is wrong
    in either direction depending on the topology: a non-loopback bind reached
    over plain HTTP (the tailnet case) derives `Secure` and the browser then
    refuses to store the cookie, while a TLS-terminating proxy in front of a
    loopback bind derives plain and drops `Secure` on a genuinely HTTPS
    deployment. So a derived value gets the matching remedy; an explicit value
    is the operator's own statement about the transport and is only reported.
    """
    secure = resolve_cookie_secure(settings)
    value = str(secure).lower()
    if settings.web.cookie_secure is not None:
        flags = (
            "session cookie marked Secure, named __Host-dudamel_session"
            if secure
            else "session cookie not marked Secure"
        )
        return True, f"{value} (explicit) — {flags}"
    if secure:
        # A warning, not a failure: doctor cannot tell a plain-HTTP tailnet
        # (login loops) from a TLS proxy in front of this bind (correct as
        # derived), and a heuristic that guesses wrong must not fail the run.
        return "warn", (
            f"true (derived from non-loopback bind host {settings.web.host!r}) — "
            "dashboard URLs above are http://, and a browser will not store a Secure "
            "cookie on a plain-HTTP origin (the login page just loops); for plain-HTTP "
            "tailnet access set `cookie_secure = false` under [web] in dudamel.toml"
        )
    return True, (
        f"false (derived from loopback bind host {settings.web.host!r}) — correct for the "
        "http:// dashboard URLs above; if a TLS-terminating proxy fronts this bind, set "
        "`cookie_secure = true` under [web] in dudamel.toml"
    )


def _check_tailscale(settings: Settings) -> str:
    dashboard = f"http://{settings.web.host}:{settings.web.port}"
    try:
        proc = subprocess.run(
            ["tailscale", "status", "--json"], capture_output=True, timeout=2, text=True
        )
    except (OSError, subprocess.TimeoutExpired):
        proc = None
    if proc is not None and proc.returncode == 0:
        addr = None
        with contextlib.suppress(json.JSONDecodeError, AttributeError, KeyError):
            self_node = json.loads(proc.stdout).get("Self", {})
            dns_name = (self_node.get("DNSName") or "").rstrip(".")
            ips = self_node.get("TailscaleIPs") or []
            addr = dns_name or (ips[0] if ips else None)
        if addr:
            detail = f"connected — dashboard: http://{addr}:{settings.web.port}"
            return _line(True, "tailscale", detail)
        return _line(True, "tailscale", f"connected — dashboard: {dashboard}")
    if Path("/Applications/Tailscale.app").exists():
        detail = f"installed but not running — dashboard: {dashboard} (loopback only)"
        return _line("info", "tailscale", detail)
    return _line("info", "tailscale", f"not detected — dashboard: {dashboard} (loopback only)")


def _render_tool_table(orchestrator: Orchestrator) -> str:
    tools = sorted(orchestrator.registry.tools.values(), key=lambda t: t.name)
    if not tools:
        return "no tools registered"
    # `action` is `ACTION_LABEL_MAX` + 2, so an at-cap label cannot push
    # `origin` out of alignment -- the contract caps the label, and a column
    # narrower than the cap would misalign exactly the rows this column was
    # added to show.
    header = (
        f"{'name':<28}{'read_only':<12}{'confirm':<10}{'external':<10}"
        f"{'action':<{ACTION_LABEL_MAX + 2}}{'origin':<8}"
    )
    rows = [header, "-" * len(header)]
    rows.extend(
        f"{t.name:<28}{str(t.read_only):<12}{str(t.confirm):<10}{str(t.external):<10}"
        f"{(t.action or '-'):<{ACTION_LABEL_MAX + 2}}{t.origin:<8}"
        for t in tools
    )
    return "\n".join(rows)


# A module-level `name = App(...)` -- what makes a file in apps/ an app rather
# than a helper. Matched on source, not by importing, because the files that
# matter here are exactly the ones nothing imports.
_APP_DEFINITION = re.compile(r"^\w+\s*=\s*App\(", re.MULTILINE)


def _unregistered_app_modules(project_dir: Path, orchestrator: Orchestrator) -> list[str]:
    """Modules under `<project>/apps/` that define an `App` the project's
    orchestrator does not contain.

    An app dropped into `apps/` but never added to `assistant.py` is invisible:
    it imports nowhere, registers nothing, and `app resolution` happily reports
    `0 enabled, 0 error(s)`. This is the line that says it is sitting there.

    Judged by identity, after `assistant.py` was imported (which is when every
    module it uses landed in `sys.modules`): a module counts as registered when
    one of its `App` objects is one of the orchestrator's. A file that was never
    imported cannot be registered, whatever its name.
    """
    apps_dir = project_dir / "apps"
    if not apps_dir.is_dir():
        return []
    registered = list(orchestrator.registry.apps.values())
    found: list[str] = []
    for path in sorted(apps_dir.iterdir()):
        if path.name.startswith("_"):
            continue
        if path.is_dir():
            source, module_name = path / "__init__.py", f"apps.{path.name}"
        elif path.suffix == ".py":
            source, module_name = path, f"apps.{path.stem}"
        else:
            continue
        try:
            if not _APP_DEFINITION.search(source.read_text()):
                continue
        except OSError:
            continue
        module = sys.modules.get(module_name)
        if module is not None and any(
            candidate is app for candidate in vars(module).values() for app in registered
        ):
            continue
        found.append(module_name.removeprefix("apps."))
    return found


def _load_orchestrator_for_diagnosis(
    project_dir: Path, *, debug: bool = False
) -> tuple[Orchestrator, str | None]:
    """The project's orchestrator, or an empty stand-in plus the message that
    explains why there isn't one.

    `doctor` and `apps list` describe projects that may not import at all, so a
    missing or broken `assistant.py` has to become a reported line rather than
    an abort. `SystemExit` is named alongside `Exception` because it is not
    one: a module calling `sys.exit()` at import would otherwise take the whole
    diagnosis down. `KeyboardInterrupt` is deliberately still allowed through.

    `--debug` opts back out of all of that. Swallowing the traceback is right
    by default (the other checks still have to run) but it leaves an operator
    debugging their own broken `assistant.py` with a single `repr` and no way
    to ask for more; `debug` re-raises so `main` can print the real stack.
    """
    try:
        return _load_orchestrator(project_dir, _DEFAULT_MODULE), None
    except DudamelError as e:
        if debug:
            raise
        return Orchestrator(apps=[]), str(e)
    except (Exception, SystemExit) as e:
        if debug:
            raise
        # The project's own code raised; report it verbatim instead of dying
        # before the unrelated checks have had a chance to run.
        return Orchestrator(apps=[]), f"{_DEFAULT_MODULE}.py raised on import: {e!r}"


def cmd_doctor(args: argparse.Namespace) -> int:
    project_dir = Path.cwd()
    # Without a dudamel.toml this is not a project, and every check below would
    # report on defaults for a directory `dudamel run` could never start in --
    # a cascade of unrelated lines that buries the one thing wrong. Say it once.
    if not (project_dir / "dudamel.toml").exists():
        print(
            f"dudamel: no dudamel.toml in {project_dir} -- this is not a dudamel project. "
            "Create one with `dudamel new <name>`, or cd into an existing project.",
            file=sys.stderr,
        )
        return 1
    _load_dotenv_into_environ(project_dir)
    settings = Settings.load(project_dir)
    # Resolved up front, non-strict: the migration check below needs the
    # enabled suite apps' lanes, and a broken [apps.*] block must not stop
    # doctor reaching any of its other checks.
    orchestrator, import_error = _load_orchestrator_for_diagnosis(project_dir, debug=args.debug)
    resolution = resolve_apps(orchestrator, settings, strict=False)

    # Every ✗ (and only a ✗) is a failure that makes doctor exit 1; see `Status`.
    # Lines are collected through `emit` so the verdict cannot drift from what
    # was printed.
    failed = False
    lines: list[str] = []

    def emit(status: Status, label: str, detail: str) -> None:
        nonlocal failed
        failed = failed or status is False
        lines.append(_line(status, label, detail))

    ok, detail = _check_db_connect(settings.database_url)
    emit(ok, "database connection", detail)

    ok, detail = _check_core_migrations(settings.database_url, auto_migrate=settings.auto_migrate)
    emit(ok, "core migrations", detail)

    # `settings.project_dir`, not the cwd: an explicit `project_dir` in
    # dudamel.toml wins, and the runtime resolves the project's own migration
    # lane from exactly that. Reading the cwd here would report on a different
    # directory than the one the startup gate gates on.
    ok, detail = _check_app_migrations_dir(
        settings.project_dir, any_local_apps=bool(resolution.local_apps)
    )
    emit(ok, "app migrations dir", detail)

    emit(
        not resolution.errors,
        "app resolution",
        f"{len(resolution.apps)} enabled, {len(resolution.errors)} error(s)",
    )
    for err in resolution.errors:
        emit(False, f"  app {err.app}", err.message)

    ok, detail = _check_pending_migrations(
        settings.database_url,
        settings.project_dir,
        resolution.suite_lanes,
        auto_migrate=settings.auto_migrate,
    )
    emit(ok, "pending migrations", detail)

    if settings.llm_tiers:
        for name, cfg in settings.llm_tiers.items():
            ok, detail = _check_tier(cfg)
            emit(ok, f"llm tier {name!r}", detail)
        if args.probe_tools:
            for name, cfg in settings.llm_tiers.items():
                ok, detail = _probe_tier_tool_calling(name, cfg)
                emit(ok, f"llm tier {name!r} tool calling", detail)
    else:
        # `run` starts without a tier (the assistant just cannot answer), so
        # this is a state to know about, not a fault.
        emit("info", "llm tiers", "none configured in dudamel.toml")

    ok, detail = _check_telegram(settings)
    emit(ok, "telegram", detail)

    ok, detail = _check_web_token(settings)
    emit(ok, "web token", detail)

    ok, detail = _check_timezone(settings)
    emit(ok, "timezone", detail)
    boundary = _check_day_boundary(settings, resolution)
    if boundary is not None:
        emit(boundary[0], "day boundary", boundary[1])

    lines.append(_check_tailscale(settings))  # informational only, never a failure

    # Last, so it reads against the dashboard URL the tailscale line just
    # printed -- that URL's scheme is what the cookie posture is judged by.
    ok, detail = _check_cookie_secure(settings)
    emit(ok, "cookie_secure", detail)

    print("\n".join(lines))
    print()

    configured = [wid for section in settings.home.section for wid in section.widgets]

    if import_error is not None:
        failed = True  # `dudamel run` imports the same module and stops here
        print(_line(False, f"app import ({_DEFAULT_MODULE})", import_error))
    else:
        # Rendered from the RESOLVED apps, not from the project's own object:
        # doctor announces `app resolution: N enabled` two lines above, and an
        # operator reads this table to decide what can run unconfirmed. A suite
        # app enabled purely in dudamel.toml is in `resolution.apps` and not in
        # the project's registry, so rendering the latter would announce the
        # app and then omit its tools from the one table that lists their
        # safety flags. `mcp` is carried over because it belongs to the
        # project's orchestrator, not to any app -- same reconstruction
        # `cmd_run` hands to `serve`.
        #
        # Guarded, because that reconstruction is also the FIRST place the
        # cross-app collision guards in `Registry.__init__` run: `resolve_apps`
        # validates each app alone, so two apps that each declare the same tool
        # name resolve cleanly and collide only here. That is a configuration
        # `dudamel run` also refuses -- precisely when an operator reaches for
        # doctor -- so it has to be a reported line, not the end of the report.
        try:
            table = _render_tool_table(Orchestrator(apps=resolution.apps, mcp=orchestrator.mcp))
        except Exception as e:
            # Same escape hatch as the import above: the one-line collision
            # message names the tool but not the apps' import sites.
            if args.debug:
                raise
            failed = True  # the same collision stops `dudamel run`
            table = _line(False, "tool table", str(e))
        print(table)
        for name in _unregistered_app_modules(settings.project_dir, orchestrator):
            print(
                _line(
                    "warn",
                    "unregistered app",
                    f"apps/{name} defines an App that assistant.py does not register -- "
                    "add it to `Orchestrator(apps=[...])`, or it never runs",
                )
            )
        # `doctor` never starts the orchestrator, so MCP-mounted tools (only
        # discovered by actually connecting to each server -- see
        # mcp_mount.py) aren't in the table above yet; this makes that gap
        # visible instead of silently under-reporting the tool-safety table.
        if orchestrator.mcp:
            n = len(orchestrator.mcp)
            print(
                f"ℹ {n} MCP server(s) configured — tools mount at run time; "
                "safety flags visible then"
            )

        # Homescreen layout, first half. Both layout conditions degrade
        # silently at render time (an unknown id is skipped, a repeat renders
        # once at its first mention), which is the right runtime behaviour and
        # exactly why they need a voice here: otherwise a typo in
        # `[[home.section]]` is indistinguishable from a widget that never ran.
        # Reported as a warning, never fatal -- doctor's exit code says nothing
        # about a layout: `run` starts fine and the dashboard just skips the id.
        #
        # Inside the `else`, and for the same reason the tool table is: with no
        # importable assistant there are no resolved apps, so EVERY configured
        # id would be called dead. The operator would then go edit dudamel.toml
        # chasing a problem that does not exist, in precisely the situation
        # doctor exists to disambiguate. The import line above is the finding.
        #
        # Judged against the RESOLVED apps, not the project's own registry:
        # `resolution.apps` is the exact set the dashboard renders from, so it
        # is right in both directions -- a suite app enabled purely in
        # dudamel.toml is resolved but not in the registry (its ids are live),
        # and a local app switched off with `enabled = false` is in the
        # registry but not resolved (its ids are dead). Read from each app
        # directly rather than through a reconstructed Registry, which can
        # raise on a cross-app collision the table above already reports.
        registered = {w.qualified_id for app in resolution.apps for w in app.widgets.values()}
        for wid in configured:
            if wid not in registered:
                print(_line("warn", "home layout", f"{wid} is not a registered widget"))

    # Second half, unguarded: this one reads config against itself and needs no
    # registry at all, so a broken assistant.py does not make it unanswerable.
    seen: set[str] = set()
    for wid in configured:
        if wid in seen:
            print(_line("warn", "home layout", f"{wid} is listed more than once"))
        seen.add(wid)
    return 1 if failed else 0


# --- apps list ---------------------------------------------------------------

# The lane column holds ONE kind of thing: the migration state of the lane that
# app's tables live in. It is filled in only for apps that actually resolved --
# a disabled, uninstallable or errored app is not going to run, is described
# from registry metadata alone (a suite app is never even imported), and its
# lane is never compared against the database, so a status there would be a
# claim the command did not check. Those rows get this dash instead.
_NO_LANE = "—"

_APPS_LIST_HEADERS = ("name", "origin", "state", "lane", "notes")


def _app_state(name: str, *, enabled: bool, resolved: set[str], errored: set[str | None]) -> str:
    if name in errored:
        return "error"
    if name in resolved:
        return "enabled"
    # Enabled but neither resolved nor blamed by name: not running either way.
    return "disabled" if not enabled else "error"


def _lane_status(db_url: str, is_pending: Callable[[], bool]) -> str:
    """One lane's migration state as a column value.

    `is_pending` is the same comparison the startup gate makes, per lane
    (`migrate.suite_lane_pending` for a suite app, `migrate.project_lane_pending`
    for the shared lane every local app's tables live in). Never raises and
    never connects to a database that does not exist yet: `apps list` describes
    a configuration, so an unreadable lane is a value in the table, not a failed
    command.
    """
    path = _sqlite_file_path(db_url)
    if path is not None and not path.exists():
        return "no db"
    try:
        return "pending" if is_pending() else "at head"
    except Exception:
        return "unknown"


def _render_apps_table(rows: Sequence[tuple[str, str, str, str, str]]) -> str:
    # Every column but the last (free-text notes) is padded to its widest cell.
    widths = [max(len(row[i]) for row in (_APPS_LIST_HEADERS, *rows)) for i in range(4)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths) + "  {}"
    rendered = [fmt.format(*row) for row in (_APPS_LIST_HEADERS, *rows)]
    # Measured across every rendered line: the notes column is unpadded, so a
    # rule sized from the header alone under-runs the rows it sits above.
    rule = "-" * max(len(line) for line in rendered)
    return "\n".join([rendered[0], rule, *rendered[1:]])


def cmd_apps_list(args: argparse.Namespace) -> int:
    project_dir = Path.cwd()
    _load_dotenv_into_environ(project_dir)
    settings = Settings.load(project_dir)
    orchestrator, import_error = _load_orchestrator_for_diagnosis(project_dir, debug=args.debug)
    # Non-strict: listing a broken configuration is the whole point.
    resolution = resolve_apps(orchestrator, settings, strict=False)
    resolved = {app.name for app in resolution.apps}
    errored: set[str | None] = {err.app for err in resolution.errors}
    lanes = dict(resolution.suite_lanes)

    rows: list[tuple[str, str, str, str, str]] = []
    # Read through the module: tests (and any in-process override) replace the
    # attribute itself, exactly as the resolver reads it.
    suite_apps = suite.SUITE_APPS
    for name, entry in sorted(suite_apps.items()):
        note = entry.summary
        missing = missing_requirements(entry)
        if missing:
            note += f" — needs pip install dudamel[{entry.extra or name}]"
        rows.append(
            (
                name,
                "suite",
                _app_state(
                    name,
                    enabled=_suite_app_enabled(settings.apps, name),
                    resolved=resolved,
                    errored=errored,
                ),
                _lane_status(
                    settings.database_url,
                    partial(suite_lane_pending, settings.database_url, name, lanes[name]),
                )
                if name in lanes
                else _NO_LANE,
                note,
            )
        )
    # One shared lane for every local app -- `migrations/` under project_dir --
    # so its state is read once and repeated per row rather than per app.
    project_lane = _lane_status(
        settings.database_url,
        partial(project_lane_pending, settings.database_url, settings.project_dir),
    )
    for name, app in sorted(orchestrator.registry.apps.items()):
        if name in suite_apps:
            continue  # the name collision is reported as an error below
        # Opposite default from a suite app, matching the resolver: a local app
        # is registered in Python, so it runs unless config switches it off.
        enabled = bool(settings.apps.get(name, {}).get("enabled", True))
        state = _app_state(name, enabled=enabled, resolved=resolved, errored=errored)
        lane = project_lane if name in resolved else _NO_LANE
        rows.append((name, "local", state, lane, app.description))

    if rows:
        print(_render_apps_table(rows))
    else:
        print("no apps — none enabled from the suite, none registered in this project")

    if import_error is not None:
        print()
        print(_line(False, f"app import ({_DEFAULT_MODULE})", import_error))
    if resolution.errors:
        print()
        for err in resolution.errors:
            print(_line(False, f"app {err.app}", err.message))
    return 0


# --- token rotate ------------------------------------------------------------


def _rewrite_env_var(path: Path, key: str, value: str) -> None:
    """Rewrite `key=...` in place, preserving every other line byte-for-byte
    (comments, blank lines, unrelated vars) -- appends the line if `key`
    isn't present yet rather than touching anything else."""
    lines = path.read_text().splitlines(keepends=True)
    new_line = f"{key}={value}\n"
    prefix = f"{key}="
    replaced = False
    out: list[str] = []
    for line in lines:
        if not replaced and line.lstrip().startswith(prefix):
            out.append(new_line)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        if out and not out[-1].endswith("\n"):
            out[-1] += "\n"
        out.append(new_line)
    path.write_text("".join(out))


def cmd_token_rotate(args: argparse.Namespace) -> int:
    project_dir = Path.cwd()
    settings = Settings.load(project_dir)
    env_path = project_dir / ".env"
    if not env_path.exists():
        raise CliError(f"{env_path} not found — is {project_dir} a dudamel project?")
    _rewrite_env_var(env_path, settings.web.token_env, secrets.token_urlsafe(32))
    os.chmod(env_path, 0o600)
    print(f"rotated {settings.web.token_env} in {env_path}")
    # Not revoked yet: the web auth compares against the environment captured
    # when the process started, and this command only rewrites the file. Said
    # plainly because "rotated" reads as "the old token is dead", which is the
    # reason people rotate.
    print(
        "A running dudamel instance keeps accepting the OLD token until it is restarted.\n"
        "Restart it to revoke the old token: `systemctl --user restart dudamel.service` "
        "(Linux unit),\n"
        "`launchctl unload` then `launchctl load ~/Library/LaunchAgents/dudamel.plist` "
        "(macOS),\n"
        "or stop and re-run `dudamel run`."
    )
    return 0


# --- argument parsing / dispatch ---------------------------------------------


def _debug_parent() -> argparse.ArgumentParser:
    """`--debug` is attached to every LEAF command (`dudamel run --debug`,
    `dudamel db migrate ... --debug`, etc.) via `parents=`, never to the
    top-level parser: `add_subparsers()` hands each subcommand a brand new
    `Namespace` and then copies its attributes back onto the outer one
    (see `argparse._SubParsersAction.__call__`), which would silently
    reset an outer `--debug=True` back to its subparser-local default of
    `False` if both levels declared the same flag -- so exactly one level
    (the leaf that actually runs) may ever declare it."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--debug", action="store_true", help="show a full traceback instead of a one-line error"
    )
    return parent


def _build_parser() -> argparse.ArgumentParser:
    debug = _debug_parent()
    parser = argparse.ArgumentParser(
        prog="dudamel", description="Run and manage a dudamel project."
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p_new = sub.add_parser("new", help="scaffold a new dudamel project", parents=[debug])
    p_new.add_argument("name", help="directory to create — must not already contain files")
    p_new.set_defaults(handler=cmd_new)

    p_run = sub.add_parser(
        "run", help="run this project (web dashboard + optional Telegram)", parents=[debug]
    )
    p_run.add_argument(
        "module",
        nargs="?",
        default=_DEFAULT_MODULE,
        help=f"module to import from the current directory (default: {_DEFAULT_MODULE!r})",
    )
    p_run.set_defaults(handler=cmd_run)

    p_db = sub.add_parser("db", help="database commands")
    db_sub = p_db.add_subparsers(dest="db_command", required=True, metavar="command")
    p_migrate = db_sub.add_parser(
        "migrate", help="generate and apply an app migration", parents=[debug]
    )
    p_migrate.add_argument("-m", "--message", required=True, help="migration message")
    p_migrate.add_argument(
        "--allow-destructive", action="store_true", help="allow drop-table/drop-column operations"
    )
    p_migrate.set_defaults(handler=cmd_db_migrate)

    p_doctor = sub.add_parser(
        "doctor", help="diagnose this project's configuration", parents=[debug]
    )
    p_doctor.add_argument(
        "--probe-tools",
        action="store_true",
        help="probe each llm tier for native tool calling (spends real tokens; off by default)",
    )
    p_doctor.set_defaults(handler=cmd_doctor)

    p_apps = sub.add_parser("apps", help="inspect the configured app suite")
    apps_sub = p_apps.add_subparsers(dest="apps_command", required=True, metavar="command")
    apps_sub.add_parser(
        "list", help="list suite and local apps with their state", parents=[debug]
    ).set_defaults(handler=cmd_apps_list)

    p_token = sub.add_parser("token", help="manage the web dashboard token")
    token_sub = p_token.add_subparsers(dest="token_command", required=True, metavar="command")
    token_sub.add_parser(
        "rotate", help="rotate the web dashboard token in .env", parents=[debug]
    ).set_defaults(handler=cmd_token_rotate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except Exception as e:
        if args.debug:
            raise
        if isinstance(e, DudamelError):
            print(f"dudamel: {e}", file=sys.stderr)
        else:
            print(f"dudamel: error: {e}", file=sys.stderr)
            print("(re-run with --debug for a full traceback)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
