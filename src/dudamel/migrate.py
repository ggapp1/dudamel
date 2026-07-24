from __future__ import annotations

import shutil
from importlib.resources import files
from pathlib import Path

from alembic import command
from alembic.config import Config


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
