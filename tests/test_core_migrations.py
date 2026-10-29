from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

from dudamel.exceptions import DudamelError
from dudamel.migrate import _sqlite_path, sync_url, upgrade_core

CORE_TABLES = {
    "conversations",
    "messages",
    "activity",
    "job_runs",
    "pending_confirmations",
    "llm_calls",
    "summaries",
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


def test_upgrade_core_is_idempotent_and_a_noop_takes_no_backup(tmp_path: Path):
    url = f"sqlite+aiosqlite:///{tmp_path}/core.db"
    upgrade_core(url)
    upgrade_core(url)  # second run: no error, and nothing was pending
    assert list(tmp_path.glob("core.db.bak*")) == []


def test_sqlite_path_strips_query_string(tmp_path: Path):
    db_file = tmp_path / "app.db"
    url = f"sqlite+aiosqlite:///{db_file}?check_same_thread=false"
    assert _sqlite_path(url) == db_file


def test_sqlite_path_returns_none_for_non_sqlite_url():
    assert _sqlite_path("postgresql+asyncpg://u@h/d") is None


@pytest.mark.parametrize(
    "url",
    ["sqlite+aiosqlite:///:memory:", "sqlite+aiosqlite://", "sqlite://"],
)
def test_sqlite_path_rejects_pathless_urls(url: str):
    with pytest.raises(DudamelError, match="in-memory"):
        _sqlite_path(url)


def test_upgrade_core_rejects_memory_url():
    """A ':memory:' SQLite URL has no file to back up; fail loudly with a
    clear DudamelError instead of crashing on an IndexError deep inside
    path-parsing, or silently skipping the backup."""
    with pytest.raises(DudamelError, match="in-memory"):
        upgrade_core("sqlite+aiosqlite:///:memory:")
