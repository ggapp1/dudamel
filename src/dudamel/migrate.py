from __future__ import annotations

import shutil
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

from dudamel.exceptions import DestructiveMigrationError
from dudamel.orchestrator import Orchestrator


def sync_url(db_url: str) -> str:
    return db_url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg")


def _sqlite_path(db_url: str) -> Path | None:
    if db_url.startswith("sqlite"):
        return Path(db_url.split("///", 1)[1])
    return None


def _backup_sqlite(db_url: str) -> None:
    path = _sqlite_path(db_url)
    if path is not None and path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))


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

    def include_object(obj, name, type_, reflected, compare_to) -> bool:
        if type_ == "table":
            # only tables of currently registered apps take part in the diff;
            # core tables and unregistered-app tables are invisible -> never dropped
            return name is not None and name.startswith(prefixes)
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
    finally:
        engine.dispose()

    if not migration.upgrade_ops.ops:
        return None

    destructive = _destructive_ops(migration.upgrade_ops)
    if destructive and not allow_destructive:
        raise DestructiveMigrationError(
            "migration contains destructive operations: "
            + "; ".join(destructive)
            + " — re-run with allow_destructive=True after reviewing"
        )

    script_dir = ScriptDirectory.from_config(_app_config(db_url, project_dir))
    head = script_dir.get_current_head()
    revision = rev_id()
    upgrade_code = render_python_code(
        migration.upgrade_ops,
        render_as_batch=engine.dialect.name == "sqlite",
        migration_context=mc,
    )
    body = f'''"""{message}

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
    path = mig_dir / "versions" / f"{revision}_{message.replace(' ', '_')[:40]}.py"
    path.write_text(body)
    return path


def upgrade_apps(db_url: str, project_dir: Path) -> None:
    """Apply the project's app migrations (version table alembic_version_apps)."""
    _backup_sqlite(db_url)
    command.upgrade(_app_config(db_url, project_dir), "head")
