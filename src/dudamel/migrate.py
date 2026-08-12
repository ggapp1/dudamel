from __future__ import annotations

import re
import shutil
import sqlite3
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

from alembic import command
from alembic.autogenerate import produce_migrations, render_python_code
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import ops as alembic_ops
from alembic.script import ScriptDirectory
from alembic.util import rev_id
from sqlalchemy import MetaData, create_engine

from dudamel.exceptions import DestructiveMigrationError, DudamelError
from dudamel.models_core import CoreBase
from dudamel.orchestrator import Orchestrator

_MESSAGE_RE = re.compile(r"[^a-z0-9_]+")


def sync_url(db_url: str) -> str:
    return db_url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg")


def script_heads(script_location: str) -> set[str]:
    cfg = Config()
    cfg.set_main_option("script_location", script_location)
    return set(ScriptDirectory.from_config(cfg).get_heads())


def current_heads(db_url: str, version_table: str) -> set[str]:
    engine = create_engine(sync_url(db_url))
    try:
        with engine.connect() as conn:
            mc = MigrationContext.configure(conn, opts={"version_table": version_table})
            return set(mc.get_current_heads())
    finally:
        engine.dispose()


def _sanitize_message(message: str) -> str:
    """Sanitize an arbitrary migration message for safe use in a filename and
    in a generated Python triple-quoted docstring: lowercase, collapse every
    run of characters outside [a-z0-9_] into a single "_", strip leading/
    trailing "_", and truncate to 40 chars. Without this, a message
    containing "/" could write outside the versions/ directory and one
    containing '"' could break out of the generated docstring."""
    sanitized = _MESSAGE_RE.sub("_", message.lower()).strip("_")
    return sanitized[:40] or "migration"


def _sqlite_path(db_url: str) -> Path | None:
    """Return the on-disk path for a SQLite URL, or None for a non-SQLite URL.

    Raises DudamelError for a SQLite URL with no real file path (`:memory:`
    or a path-less `sqlite://`) since there is nothing to back up. Strips any
    query string (e.g. `?check_same_thread=false`) from the path component.
    """
    if not db_url.startswith("sqlite"):
        return None
    if "///" not in db_url:
        raise DudamelError(
            f"sqlite URL {db_url!r} has no file path (in-memory database); cannot back it up"
        )
    raw_path = db_url.split("///", 1)[1].split("?", 1)[0]
    if not raw_path or raw_path == ":memory:":
        raise DudamelError(
            f"sqlite URL {db_url!r} has no file path (in-memory database); cannot back it up"
        )
    return Path(raw_path)


def _backup_sqlite(db_url: str) -> None:
    """Back up a SQLite file with the sqlite3 online backup API rather than a
    plain file copy: `Connection.backup()` goes through SQLite itself, so it
    is safe against a source database that is mid-write or in WAL mode —
    unlike `shutil.copy2`, which can copy an inconsistent snapshot of the
    main file while the real data still lives in the `-wal` sidecar."""
    path = _sqlite_path(db_url)
    if path is None or not path.exists():
        return
    backup_path = path.with_name(path.name + ".bak")
    src = sqlite3.connect(path)
    try:
        dst = sqlite3.connect(backup_path)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def upgrade_core(db_url: str) -> None:
    """Apply framework-bundled core migrations (backup first on SQLite)."""
    _backup_sqlite(db_url)
    cfg = Config()
    cfg.set_main_option("script_location", str(files("dudamel") / "migrations"))
    cfg.set_main_option("sqlalchemy.url", sync_url(db_url))
    cfg.set_main_option("version_table", "alembic_version_core")
    command.upgrade(cfg, "head")


def ensure_app_migrations(project_dir: Path) -> Path:
    """Create `<project>/migrations/` (env.py + script.py.mako + versions/) from
    the packaged app template if it doesn't already exist. Idempotent."""
    mig_dir = project_dir / "migrations"
    if not mig_dir.exists():
        mig_dir.mkdir(parents=True)
        (mig_dir / "versions").mkdir()
        template = files("dudamel") / "migrations_app_template"
        for name in ("env.py", "script.py.mako"):
            shutil.copyfile(str(template / name), mig_dir / name)
    return mig_dir


def _app_config(db_url: str, project_dir: Path) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(project_dir / "migrations"))
    cfg.set_main_option("sqlalchemy.url", sync_url(db_url))
    return cfg


def suite_version_table(app_name: str) -> str:
    """One bookkeeping table per suite app, so enabling or disabling one app
    can never disturb another's recorded revision."""
    return f"alembic_version_app_{app_name}"


def _suite_lane_config(db_url: str, app_name: str, versions_dir: Path) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(files("dudamel") / "migrations_suite_lane"))
    # Without this, Alembic splits version_locations on spaces and commas, so a
    # project path containing either would silently resolve to no revisions.
    cfg.set_main_option("path_separator", "os")
    cfg.set_main_option("version_locations", str(versions_dir))
    cfg.set_main_option("version_table", suite_version_table(app_name))
    cfg.set_main_option("sqlalchemy.url", sync_url(db_url))
    return cfg


def _lane_heads(versions_dir: Path) -> set[str]:
    """Heads of one lane's revisions. They live in `version_locations`, not in
    the shared script directory, so `script_heads` cannot read them."""
    cfg = Config()
    cfg.set_main_option("script_location", str(files("dudamel") / "migrations_suite_lane"))
    cfg.set_main_option("path_separator", "os")
    cfg.set_main_option("version_locations", str(versions_dir))
    return set(ScriptDirectory.from_config(cfg).get_heads())


def upgrade_suite_app(db_url: str, app_name: str, versions_dir: Path) -> None:
    """Bring one suite app's lane to head. No backup here -- upgrade_all takes
    a single backup before touching any lane."""
    command.upgrade(_suite_lane_config(db_url, app_name, versions_dir), "head")


def upgrade_all(db_url: str, project_dir: Path, suite_lanes: Sequence[tuple[str, Path]]) -> None:
    """Apply every suite lane in name order, then the project's own lane.

    Sequential, not atomic. A lane that fails leaves earlier lanes applied and
    stops the rest from running; the raised error names the lane, and re-running
    after the fix is safe because a lane already at head is a no-op.
    """
    _backup_sqlite(db_url)
    for app_name, versions_dir in sorted(suite_lanes):
        try:
            upgrade_suite_app(db_url, app_name, versions_dir)
        except Exception as e:
            raise DudamelError(
                f"migration lane for app {app_name!r} failed: {e}; "
                "lanes applied before it remain applied, later lanes did not run"
            ) from e
    # The project's own lane only exists once the user has generated a
    # revision; a project with no migrations/ directory has nothing to apply.
    if (project_dir / "migrations").exists():
        command.upgrade(_app_config(db_url, project_dir), "head")


def _combined_metadata(orc: Orchestrator) -> MetaData:
    """Union of every registered app's tables into one MetaData, so autogenerate
    diffs against exactly (and only) what's currently registered."""
    combined = MetaData()
    for md in orc.registry.metadatas.values():
        for table in md.tables.values():
            table.to_metadata(combined)
    return combined


def _destructive_ops(upgrade_ops: alembic_ops.UpgradeOps) -> list[str]:
    """Walk the diff tree for drop-table/drop-column ops. Alembic nests
    per-table alterations (including batch-mode ones, used on SQLite) inside
    ModifyTableOps containers, so recurse into those."""
    found: list[str] = []

    def walk(op_container) -> None:
        for op in op_container.ops:
            if isinstance(op, alembic_ops.DropTableOp):
                found.append(f"drop table {op.table_name}")
            elif isinstance(op, alembic_ops.DropColumnOp):
                found.append(f"drop column {op.table_name}.{op.column_name}")
            elif isinstance(op, alembic_ops.ModifyTableOps):
                walk(op)

    walk(upgrade_ops)
    return found


def generate_app_migration(
    orc: Orchestrator,
    db_url: str,
    message: str,
    project_dir: Path,
    allow_destructive: bool = False,
) -> Path | None:
    """Autogenerate one revision from the combined metadata of *registered*
    apps only. Core tables and unregistered-app tables never enter the diff
    (prefix allowlist via include_object) so they can never be dropped by an
    app-tier migration. Destructive ops (drop table/column) raise unless
    allow_destructive=True. Returns None when there is nothing to migrate."""
    mig_dir = ensure_app_migrations(project_dir)
    prefixes = tuple(f"{name}_" for name in orc.registry.apps)
    metadata = _combined_metadata(orc)
    core_table_names = frozenset(CoreBase.metadata.tables)

    def include_object(obj, name, type_, reflected, compare_to) -> bool:
        if type_ == "table":
            # Defense in depth, checked BEFORE the prefix allowlist below: core
            # tables and the alembic version-table namespace are never part of
            # an app-tier diff, full stop — even if some app's prefix were to
            # (mis)match a core table name, it must never enter the diff and
            # risk being dropped. Registry already refuses to register such an
            # app name, but this check does not rely on that holding true.
            if name is None or name in core_table_names or name.startswith("alembic_"):
                return False
            # only tables of currently registered apps take part in the diff;
            # unregistered-app tables are invisible -> never dropped
            return name.startswith(prefixes)
        return True

    engine = create_engine(sync_url(db_url))
    try:
        with engine.connect() as conn:
            mc = MigrationContext.configure(
                conn,
                opts={
                    "compare_type": True,
                    "include_object": include_object,
                    "target_metadata": metadata,
                },
            )
            migration = produce_migrations(mc, metadata)

            if not migration.upgrade_ops.ops:
                return None

            destructive = _destructive_ops(migration.upgrade_ops)
            if destructive and not allow_destructive:
                raise DestructiveMigrationError(
                    "migration contains destructive operations: "
                    + "; ".join(destructive)
                    + " — re-run with --allow-destructive after reviewing"
                )

            # Rendered here, inside the `with engine.connect()` block, while
            # `mc` (the MigrationContext passed as migration_context) is still
            # bound to a live connection rather than one already closed by
            # engine disposal below.
            upgrade_code = render_python_code(
                migration.upgrade_ops,
                render_as_batch=engine.dialect.name == "sqlite",
                migration_context=mc,
            )
    finally:
        engine.dispose()

    script_dir = ScriptDirectory.from_config(_app_config(db_url, project_dir))
    head = script_dir.get_current_head()
    revision = rev_id()
    safe_message = _sanitize_message(message)
    body = f'''"""{safe_message}

Revision ID: {revision}
Revises: {head}
"""
import sqlalchemy as sa
from alembic import op

revision = {revision!r}
down_revision = {head!r}
branch_labels = None
depends_on = None


def upgrade() -> None:
    {upgrade_code}


def downgrade() -> None:
    raise NotImplementedError("app migrations are forward-only in dudamel v1")
'''
    path = mig_dir / "versions" / f"{revision}_{safe_message}.py"
    path.write_text(body)
    return path


def upgrade_apps(db_url: str, project_dir: Path) -> None:
    """Apply the project's app migrations (version table alembic_version_apps)."""
    _backup_sqlite(db_url)
    command.upgrade(_app_config(db_url, project_dir), "head")


def pending_migrations(
    db_url: str,
    project_dir: Path,
    suite_lanes: Sequence[tuple[str, Path]] = (),
) -> list[str]:
    """Which migration tiers are behind their scripts' heads.

    Reported in apply order: core, then each suite lane, then the project's own
    lane. Empty means the schema is current and starting up will not change it.
    Used both by `dudamel doctor` and by the startup gate, so the two can
    never disagree about what "up to date" means.
    """
    pending: list[str] = []

    core_scripts = script_heads(str(files("dudamel") / "migrations"))
    if core_scripts and current_heads(db_url, "alembic_version_core") != core_scripts:
        pending.append("core schema is behind head")

    for app_name, versions_dir in sorted(suite_lanes):
        lane_scripts = _lane_heads(versions_dir)
        if lane_scripts and current_heads(db_url, suite_version_table(app_name)) != lane_scripts:
            pending.append(f"app {app_name!r} schema is behind head")

    migrations_dir = project_dir / "migrations"
    if migrations_dir.exists():
        app_scripts = script_heads(str(migrations_dir))
        # Version table matches migrations_app_template/env.py's
        # context.configure(version_table="alembic_version_apps"), the same
        # table upgrade_apps() writes to via _app_config -> command.upgrade.
        if app_scripts and current_heads(db_url, "alembic_version_apps") != app_scripts:
            pending.append("app schema is behind head")

    return pending
