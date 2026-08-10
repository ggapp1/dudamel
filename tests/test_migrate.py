from pathlib import Path

from dudamel.migrate import pending_migrations, upgrade_core


def test_pending_migrations_empty_when_at_head(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'd.db'}"
    upgrade_core(db_url)
    assert pending_migrations(db_url, tmp_path) == []


def test_pending_migrations_reports_core_when_never_applied(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'd.db'}"
    pending = pending_migrations(db_url, tmp_path)
    assert pending
    assert any("core" in p for p in pending)
