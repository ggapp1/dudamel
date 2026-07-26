"""`dudamel` command-line entry point (Plan 4 Task 3) — the product's front
door: `dudamel new/run/db migrate/doctor/token rotate`.

argparse only (Global Constraints: no click/typer). Every command below
operates on the project in the CURRENT working directory (`new` is the one
exception — it creates a project elsewhere) — this mirrors the scaffold's
own README, which always tells you to `cd` into the project first.

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
import os
import secrets
import shutil
import subprocess
import sys
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from types import ModuleType

import httpx
from alembic.config import Config as AlembicConfig
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from dudamel.config import Settings, TierConfig
from dudamel.exceptions import DudamelError
from dudamel.interfaces.telegram import resolve_token as resolve_telegram_token
from dudamel.migrate import (
    ensure_app_migrations,
    generate_app_migration,
    sync_url,
    upgrade_apps,
    upgrade_core,
)
from dudamel.orchestrator import Orchestrator
from dudamel.serve import serve
from dudamel.web.auth import resolve_token as resolve_web_token

__all__ = ["main"]

_DEFAULT_MODULE = "assistant"


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
    print("  uv run dudamel db migrate -m init")
    print("  uv run dudamel run")
    return 0


# --- run -----------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    project_dir = Path.cwd()
    _load_dotenv_into_environ(project_dir)
    settings = Settings.load(project_dir)
    orchestrator = _load_orchestrator(project_dir, args.module)
    asyncio.run(serve(orchestrator, settings))
    return 0


# --- db migrate ------------------------------------------------------------


def cmd_db_migrate(args: argparse.Namespace) -> int:
    project_dir = Path.cwd()
    settings = Settings.load(project_dir)
    orchestrator = _load_orchestrator(project_dir, _DEFAULT_MODULE)
    # Apply core migrations first to ensure schema is ready for app autogenerate
    upgrade_core(settings.database_url)
    path = generate_app_migration(
        orchestrator,
        settings.database_url,
        args.message,
        project_dir,
        allow_destructive=args.allow_destructive,
    )
    # Always bring the project db to head, whether or not this call produced
    # a new revision -- a previously-generated-but-unapplied migration file
    # must not require a second `db migrate` invocation to take effect.
    upgrade_apps(settings.database_url, project_dir)
    print(str(path) if path is not None else "no changes")
    return 0


# --- doctor ------------------------------------------------------------------


def _line(ok: bool, label: str, detail: str) -> str:
    return f"{'✓' if ok else '✗'} {label}: {detail}"


def _script_heads(script_location: str) -> set[str]:
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", script_location)
    return set(ScriptDirectory.from_config(cfg).get_heads())


def _current_heads(db_url: str, version_table: str) -> set[str]:
    engine = create_engine(sync_url(db_url))
    try:
        with engine.connect() as conn:
            mc = MigrationContext.configure(conn, opts={"version_table": version_table})
            return set(mc.get_current_heads())
    finally:
        engine.dispose()


def _check_db_connect(db_url: str) -> tuple[bool, str]:
    # For SQLite, check if the database file exists before trying to connect
    if db_url.startswith("sqlite"):
        if "///" in db_url:
            raw_path = db_url.split("///", 1)[1].split("?", 1)[0]
            if raw_path and raw_path != ":memory:":
                path = Path(raw_path)
                if not path.exists():
                    return False, "not created yet (run `dudamel run` first)"
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


def _check_core_migrations(db_url: str) -> tuple[bool, str]:
    # For SQLite, check if database file exists before trying to connect
    # (avoid creating an empty database file when just running doctor)
    if db_url.startswith("sqlite"):
        if "///" in db_url:
            raw_path = db_url.split("///", 1)[1].split("?", 1)[0]
            if raw_path and raw_path != ":memory:":
                path = Path(raw_path)
                if not path.exists():
                    return (
                        False,
                        "not yet applied — run `dudamel run` or `dudamel db migrate -m <msg>` once",
                    )
    try:
        script_heads = _script_heads(str(files("dudamel") / "migrations"))
        current = _current_heads(db_url, "alembic_version_core")
    except Exception as e:
        return False, f"could not read core migration state ({e})"
    if not script_heads:
        return False, "no core migration scripts found (packaging problem)"
    if current == script_heads:
        return True, "at head"
    if not current:
        return False, "not yet applied — run `dudamel run` or `dudamel db migrate -m <msg>` once"
    return False, f"behind head ({sorted(current)} != {sorted(script_heads)})"


def _check_app_migrations_dir(project_dir: Path) -> tuple[bool, str]:
    mig_dir = project_dir / "migrations"
    if not mig_dir.exists():
        return False, "migrations/ not found — run `dudamel db migrate -m init` to create it"
    revisions = list((mig_dir / "versions").glob("*.py")) if (mig_dir / "versions").exists() else []
    if not revisions:
        return True, "present, no revisions yet — run `dudamel db migrate -m init`"
    return True, f"present ({len(revisions)} revision{'s' if len(revisions) != 1 else ''})"


def _check_tier(cfg: TierConfig) -> tuple[bool, str]:
    if cfg.provider == "openai-compatible":
        if not cfg.base_url:
            return False, "no base_url configured"
        url = cfg.base_url.rstrip("/") + "/models"
        try:
            resp = httpx.get(url, timeout=2.0)
        except httpx.HTTPError as e:
            return False, f"{url} unreachable ({e.__class__.__name__})"
        if resp.status_code < 400:
            return True, f"{url} reachable"
        return False, f"{url} returned HTTP {resp.status_code}"
    if cfg.provider == "anthropic":
        env = cfg.api_key_env or "ANTHROPIC_API_KEY"
        if os.environ.get(env):
            return True, f"{env} is set"
        return False, f"{env} is not set"
    return True, "fake provider (tests only)"


def _check_telegram(settings: Settings) -> tuple[bool, str]:
    token = resolve_telegram_token(settings)
    if not token:
        return False, f"{settings.telegram.token_env} not set — disabled (optional)"
    n = len(settings.telegram.allowed_user_ids)
    groups = "allowed" if settings.telegram.allow_groups else "blocked"
    return True, f"token configured, {n} allowed user id(s), groups {groups}"


def _check_web_token(settings: Settings) -> tuple[bool, str]:
    token = resolve_web_token(settings)
    if not token:
        return False, f"{settings.web.token_env} not set — dashboard login impossible"
    return True, f"{settings.web.token_env} is set"


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
        return _line(False, "tailscale", detail)
    return _line(False, "tailscale", f"not detected — dashboard: {dashboard} (loopback only)")


def _render_tool_table(orchestrator: Orchestrator) -> str:
    tools = sorted(orchestrator.registry.tools.values(), key=lambda t: t.name)
    if not tools:
        return "no tools registered"
    header = f"{'name':<28}{'read_only':<12}{'confirm':<10}{'origin':<8}"
    rows = [header, "-" * len(header)]
    rows.extend(
        f"{t.name:<28}{str(t.read_only):<12}{str(t.confirm):<10}{t.origin:<8}" for t in tools
    )
    return "\n".join(rows)


def cmd_doctor(args: argparse.Namespace) -> int:
    project_dir = Path.cwd()
    _load_dotenv_into_environ(project_dir)
    settings = Settings.load(project_dir)

    ok, detail = _check_db_connect(settings.database_url)
    lines = [_line(ok, "database connection", detail)]

    ok, detail = _check_core_migrations(settings.database_url)
    lines.append(_line(ok, "core migrations", detail))

    ok, detail = _check_app_migrations_dir(project_dir)
    lines.append(_line(ok, "app migrations dir", detail))

    if settings.llm_tiers:
        for name, cfg in settings.llm_tiers.items():
            ok, detail = _check_tier(cfg)
            lines.append(_line(ok, f"llm tier {name!r}", detail))
    else:
        lines.append(_line(False, "llm tiers", "none configured in dudamel.toml"))

    ok, detail = _check_telegram(settings)
    lines.append(_line(ok, "telegram", detail))

    ok, detail = _check_web_token(settings)
    lines.append(_line(ok, "web token", detail))

    lines.append(_check_tailscale(settings))

    print("\n".join(lines))
    print()

    try:
        orchestrator = _load_orchestrator(project_dir, _DEFAULT_MODULE)
    except DudamelError as e:
        print(_line(False, f"app import ({_DEFAULT_MODULE})", str(e)))
    else:
        print(_render_tool_table(orchestrator))
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

    sub.add_parser(
        "doctor", help="diagnose this project's configuration", parents=[debug]
    ).set_defaults(handler=cmd_doctor)

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
