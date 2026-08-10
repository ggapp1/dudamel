from pathlib import Path

from dudamel import App, Orchestrator
from dudamel.migrate import (
    ensure_app_migrations,
    generate_app_migration,
    pending_migrations,
    upgrade_apps,
    upgrade_core,
)


def _make_orc() -> Orchestrator:
    app = App("blog", description="d")

    class Post(app.Model):
        title: str

    return Orchestrator(apps=[app])


def test_pending_migrations_empty_when_at_head(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'd.db'}"
    upgrade_core(db_url)
    assert pending_migrations(db_url, tmp_path) == []


def test_pending_migrations_reports_core_when_never_applied(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'd.db'}"
    pending = pending_migrations(db_url, tmp_path)
    assert pending
    assert any("core" in p for p in pending)


def test_pending_migrations_empty_when_no_migrations_dir(tmp_path: Path) -> None:
    """A fresh scaffold with no migrations/ directory at all (app tier never
    initialized) must not spuriously report the app tier as pending -- only
    the core check applies."""
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'd.db'}"
    upgrade_core(db_url)
    assert not (tmp_path / "migrations").exists()
    assert pending_migrations(db_url, tmp_path) == []


def test_pending_migrations_reports_app_when_never_applied(tmp_path: Path) -> None:
    """An app migration script exists on disk but was never applied to the
    db -- the app tier's own head comparison, independent of core, must
    catch this. Regression guard: a core-only pending_migrations would pass
    this test only by accident (it wouldn't check the app tier at all), so
    the assertion is scoped to specifically require "app" in the report."""
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'd.db'}"
    upgrade_core(db_url)
    ensure_app_migrations(tmp_path)
    generate_app_migration(_make_orc(), db_url, "add posts", tmp_path)
    pending = pending_migrations(db_url, tmp_path)
    assert any("app" in p for p in pending)


def test_pending_migrations_empty_when_app_at_head(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'd.db'}"
    upgrade_core(db_url)
    ensure_app_migrations(tmp_path)
    generate_app_migration(_make_orc(), db_url, "add posts", tmp_path)
    upgrade_apps(db_url, tmp_path)
    assert pending_migrations(db_url, tmp_path) == []
