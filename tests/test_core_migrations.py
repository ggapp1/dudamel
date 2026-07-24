from pathlib import Path

from sqlalchemy import create_engine, inspect

from dudamel.migrate import sync_url, upgrade_core

CORE_TABLES = {
    "conversations",
    "messages",
    "activity",
    "job_runs",
    "pending_confirmations",
}


def test_sync_url():
    assert sync_url("sqlite+aiosqlite:///x.db") == "sqlite:///x.db"
    assert sync_url("postgresql+asyncpg://u@h/d") == "postgresql+psycopg://u@h/d"


def test_upgrade_core_creates_tables(tmp_path: Path):
    url = f"sqlite+aiosqlite:///{tmp_path}/core.db"
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    assert CORE_TABLES <= set(insp.get_table_names())
    assert "alembic_version_core" in insp.get_table_names()


def test_upgrade_core_is_idempotent_and_backs_up(tmp_path: Path):
    url = f"sqlite+aiosqlite:///{tmp_path}/core.db"
    upgrade_core(url)
    upgrade_core(url)  # second run: no error, and a backup exists
    assert (tmp_path / "core.db.bak").exists()
