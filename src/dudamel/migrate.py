from __future__ import annotations

import re
import shutil
import sqlite3
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
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


# How many timestamped backups of one database file are kept. A backup is only
# taken when a migration is about to be applied, so five is the last five
# schema changes -- enough to step back past a bad one without the directory
# filling with copies of a database that can be large.
BACKUPS_KEPT = 5

_BACKUP_SUFFIX = ".bak-"


def _backup_name_re(db_name: str) -> re.Pattern[str]:
    """Matches exactly the backups `_backup_sqlite` makes for the database file
    called `db_name` and nothing else: another database's backups in the same
    directory, or a file the operator parked next to this one, never match."""
    return re.compile(re.escape(db_name + _BACKUP_SUFFIX) + r"\d{8}T\d{12}Z")


def _backups_of(path: Path) -> list[Path]:
    """This database's backups, oldest first (the UTC timestamp sorts lexically
    in chronological order)."""
    pattern = _backup_name_re(path.name)
    return sorted(p for p in path.parent.iterdir() if pattern.fullmatch(p.name))


def _prune_backups(path: Path) -> None:
    for old in _backups_of(path)[:-BACKUPS_KEPT]:
        old.unlink()


def _migration_state(path: Path) -> list[tuple[str, str]]:
    """Every (version table, revision) row in one SQLite file: which revision
    each tier -- core, every suite lane, the project lane -- is recorded at."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name LIKE 'alembic_version%'"
            )
        ]
        return sorted(
            (table, r[0]) for table in tables for r in conn.execute(f'SELECT * FROM "{table}"')
        )
    finally:
        conn.close()


def _newest_backup_is_current(path: Path) -> bool:
    """Whether the newest backup already holds this database's migration state.

    The crash loop this exists for: a migration that fails on every attempt,
    under the shipped unit's `Restart=always` (every 5s, forever -- below
    systemd's start-rate limit). Each start is still pending, so without this
    every start takes a backup, and within BACKUPS_KEPT restarts the prune has
    deleted the only pre-migration copy -- the very loss timestamped names were
    meant to stop. While anything is pending the startup gate refuses to run,
    so nothing writes app data in between: an unchanged migration state means
    an unchanged database, and a repeat snapshot is pure loss. A failure that
    applied some revisions before stopping DOES change the state, so that
    half-migrated copy is taken once, beside (never over) the clean one.
    """
    backups = _backups_of(path)
    if not backups:
        return False
    try:
        return _migration_state(backups[-1]) == _migration_state(path)
    except sqlite3.Error:
        # An unreadable backup is no reason to skip a fresh one.
        return False


def _backup_sqlite(db_url: str) -> Path | None:
    """Back up a SQLite file with the sqlite3 online backup API rather than a
    plain file copy: `Connection.backup()` goes through SQLite itself, so it
    is safe against a source database that is mid-write or in WAL mode --
    unlike `shutil.copy2`, which can copy an inconsistent snapshot of the
    main file while the real data still lives in the `-wal` sidecar.

    Each call writes a NEW file, `<db>.bak-<UTC timestamp>`, and never touches
    an earlier one: a fixed `<db>.bak` was overwritten by the next backup, so a
    restart after a bad partial migration re-backed-up the half-migrated state
    over the only good copy. Only the newest `BACKUPS_KEPT` are retained.
    Returns the new backup's path, or None when there is no file to back up.
    """
    path = _sqlite_path(db_url)
    if path is None or not path.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = path.with_name(f"{path.name}{_BACKUP_SUFFIX}{stamp}")
    src = sqlite3.connect(path)
    try:
        dst = sqlite3.connect(backup_path)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    _prune_backups(path)
    return backup_path


def _backup_if_pending(db_url: str, pending: Callable[[], bool], backup: bool) -> Path | None:
    """Back up only when `pending()` says something is about to be applied.
    The caller passes the same pending predicates doctor and the startup gate
    use, so "about to apply" cannot drift from what they report. The file's
    existence is checked first: the pending checks open the database, which
    would otherwise create an empty file just to decide there is nothing to
    back up. A database whose newest backup already records its migration
    state is not backed up again (see `_newest_backup_is_current`)."""
    if not backup:
        return None
    path = _sqlite_path(db_url)
    if path is None or not path.exists() or not pending():
        return None
    if _newest_backup_is_current(path):
        return None
    return _backup_sqlite(db_url)


def core_pending(db_url: str) -> bool:
    """Whether the core schema is behind the scripts shipped in the wheel."""
    core_scripts = script_heads(str(files("dudamel") / "migrations"))
    return bool(core_scripts) and current_heads(db_url, "alembic_version_core") != core_scripts


def upgrade_core(db_url: str, *, backup: bool = True) -> Path | None:
    """Apply framework-bundled core migrations (backup first on SQLite, but only
    if the core schema is actually behind). Returns the backup taken, if any."""
    taken = _backup_if_pending(db_url, lambda: core_pending(db_url), backup)
    cfg = Config()
    cfg.set_main_option("script_location", str(files("dudamel") / "migrations"))
    cfg.set_main_option("sqlalchemy.url", sync_url(db_url))
    cfg.set_main_option("version_table", "alembic_version_core")
    command.upgrade(cfg, "head")
    return taken


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


def _lane_script_config(versions_dir: Path) -> Config:
    """The half of a lane's config that locates its revisions -- shared by the
    read path (`_lane_heads`) and the write path (`_suite_lane_config`) so the
    two can never drift. A drift on the read path alone is the dangerous one:
    it reports a behind lane as up to date.

    `path_separator = "newline"` matters. A lane's `version_locations` is
    always exactly one path, and Alembic otherwise splits it: the legacy
    default on spaces and commas, "os" on `os.pathsep` (`:` on POSIX). Every
    one of those is a legal directory-name character, and a split path
    silently resolves to no revisions rather than raising -- so the lane would
    report as up to date and apply nothing. "newline" cannot occur in a path
    component in practice, and it raises no deprecation warning.
    """
    cfg = Config()
    cfg.set_main_option("script_location", str(files("dudamel") / "migrations_suite_lane"))
    cfg.set_main_option("path_separator", "newline")
    cfg.set_main_option("version_locations", str(versions_dir))
    return cfg


def _suite_lane_config(db_url: str, app_name: str, versions_dir: Path) -> Config:
    cfg = _lane_script_config(versions_dir)
    cfg.set_main_option("version_table", suite_version_table(app_name))
    cfg.set_main_option("sqlalchemy.url", sync_url(db_url))
    return cfg


def _lane_heads(versions_dir: Path) -> set[str]:
    """Heads of one lane's revisions. They live in `version_locations`, not in
    the shared script directory, so `script_heads` cannot read them."""
    return set(ScriptDirectory.from_config(_lane_script_config(versions_dir)).get_heads())


def upgrade_suite_app(db_url: str, app_name: str, versions_dir: Path) -> None:
    """Bring one suite app's lane to head. No backup here -- upgrade_all takes
    a single backup before touching any lane."""
    command.upgrade(_suite_lane_config(db_url, app_name, versions_dir), "head")


def upgrade_all(
    db_url: str,
    project_dir: Path,
    suite_lanes: Sequence[tuple[str, Path]],
    *,
    backup: bool = True,
) -> Path | None:
    """Apply every suite lane in name order, then the project's own lane.

    Sequential, not atomic. A lane that fails leaves earlier lanes applied and
    stops the rest from running; the raised error names the lane, and re-running
    after the fix is safe because a lane already at head is a no-op.

    Not atomic *within* a lane either, on SQLite: alembic's SQLite dialect sets
    transactional_ddl=False, so a lane whose second revision fails keeps the
    first one applied and recorded. That is why a rerun must resume rather than
    restart -- it picks up from the recorded revision, and the single backup
    taken below is the real undo. It is taken only when some lane is actually
    behind, and it is a new timestamped file that no later backup overwrites, so
    a crash-restart that re-runs this after a partial failure cannot replace the
    pre-migration copy with the half-migrated state. Returns the backup taken.
    """
    taken = _backup_if_pending(
        db_url,
        lambda: _lanes_pending(db_url, project_dir, suite_lanes),
        backup,
    )
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
    return taken


def upgrade_database(
    db_url: str,
    project_dir: Path,
    suite_lanes: Sequence[tuple[str, Path]],
    *,
    backup: bool = True,
) -> Path | None:
    """Bring core, every suite lane and the project lane to head under ONE
    backup, taken only if `pending_migrations` reports anything. The entry
    point for callers that apply everything (startup, `db migrate`): calling
    `upgrade_core` then `upgrade_all` separately would back up once per tier.
    `backup=False` lets a caller that already holds a pre-migration snapshot
    from this same run skip a second one. Returns the backup taken, if any."""
    taken = _backup_if_pending(
        db_url, lambda: bool(pending_migrations(db_url, project_dir, suite_lanes)), backup
    )
    upgrade_core(db_url, backup=False)
    upgrade_all(db_url, project_dir, suite_lanes, backup=False)
    return taken


def _lanes_pending(db_url: str, project_dir: Path, suite_lanes: Sequence[tuple[str, Path]]) -> bool:
    return any(
        suite_lane_pending(db_url, name, versions) for name, versions in sorted(suite_lanes)
    ) or project_lane_pending(db_url, project_dir)


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


def upgrade_apps(db_url: str, project_dir: Path, *, backup: bool = True) -> Path | None:
    """Apply the project's app migrations (version table alembic_version_apps),
    with a backup first only if the project lane is actually behind."""
    taken = _backup_if_pending(db_url, lambda: project_lane_pending(db_url, project_dir), backup)
    command.upgrade(_app_config(db_url, project_dir), "head")
    return taken


def suite_lane_pending(db_url: str, app_name: str, versions_dir: Path) -> bool:
    """Whether one suite app's lane has revisions the database has not applied.

    The per-lane comparison `pending_migrations` reports on, exposed on its own
    so `dudamel apps list`'s per-app column cannot disagree with the startup
    gate about whether that lane is behind. An app that ships no revisions at
    all is never pending.
    """
    lane_scripts = _lane_heads(versions_dir)
    if not lane_scripts:
        return False
    return current_heads(db_url, suite_version_table(app_name)) != lane_scripts


def project_lane_pending(db_url: str, project_dir: Path) -> bool:
    """Whether the project's own lane has revisions the database has not
    applied. The other half of `pending_migrations`, exposed for the same
    reason as `suite_lane_pending`: `dudamel apps list`'s lane column must not
    hold a second, drifting copy of the comparison. A project that has never
    generated a revision has no migrations/ at all and is never pending.
    """
    migrations_dir = project_dir / "migrations"
    if not migrations_dir.exists():
        return False
    app_scripts = script_heads(str(migrations_dir))
    if not app_scripts:
        return False
    # Version table matches migrations_app_template/env.py's
    # context.configure(version_table="alembic_version_apps"), the same table
    # upgrade_apps() writes to via _app_config -> command.upgrade.
    return current_heads(db_url, "alembic_version_apps") != app_scripts


def pending_migrations(
    db_url: str,
    project_dir: Path,
    suite_lanes: Sequence[tuple[str, Path]] = (),
) -> list[str]:
    """Which migration tiers are behind their scripts' heads.

    Reported in apply order: core, then each suite lane, then the project's own
    lane. Empty means the schema is current and starting up will not change it.
    Called both by `dudamel doctor` (which reports the list) and by the startup
    gate (which refuses to start on a non-empty one), so the two can never
    disagree about what "up to date" means.
    """
    pending: list[str] = []

    if core_pending(db_url):
        pending.append("core schema is behind head")

    for app_name, versions_dir in sorted(suite_lanes):
        if suite_lane_pending(db_url, app_name, versions_dir):
            pending.append(f"app {app_name!r} schema is behind head")

    if project_lane_pending(db_url, project_dir):
        pending.append("app schema is behind head")

    return pending
