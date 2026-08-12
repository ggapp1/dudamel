import pytest
from sqlalchemy import text

from dudamel import App
from dudamel.db import IN_DB_SCOPE, Database
from dudamel.exceptions import RuntimeNotBoundError


@pytest.fixture
async def db(tmp_path):
    d = Database(f"sqlite+aiosqlite:///{tmp_path}/t.db")
    async with d.engine.begin() as conn:
        await conn.execute(text("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)"))
    yield d
    await d.engine.dispose()


async def test_commit_on_clean_exit(db: Database):
    async with db.session() as s:
        await s.execute(text("INSERT INTO items (name) VALUES ('a')"))
    async with db.session() as s:
        count = (await s.execute(text("SELECT COUNT(*) FROM items"))).scalar()
    assert count == 1


async def test_rollback_on_exception(db: Database):
    with pytest.raises(RuntimeError):
        async with db.session() as s:
            await s.execute(text("INSERT INTO items (name) VALUES ('b')"))
            raise RuntimeError("boom")
    async with db.session() as s:
        count = (await s.execute(text("SELECT COUNT(*) FROM items"))).scalar()
    assert count == 0


async def test_foreign_keys_enforced(tmp_path):
    """SQLite enforces foreign keys per connection and defaults to off, so
    without the pragma every ForeignKey in models_core.py is decorative on
    the default backend -- and an FK mistake that raises on Postgres passes
    silently in a SQLite-only test run."""
    from sqlalchemy.exc import IntegrityError

    from dudamel.migrate import upgrade_core
    from dudamel.models_core import Message as MessageRow

    url = f"sqlite+aiosqlite:///{tmp_path}/fk.db"
    upgrade_core(url)
    d = Database(url)
    try:
        with pytest.raises(IntegrityError):
            async with d.session() as s:
                s.add(MessageRow(conversation_id=999_999, role="user", content={"role": "user"}))
    finally:
        await d.dispose()


async def test_wal_mode_enabled(db: Database):
    async with db.session() as s:
        mode = (await s.execute(text("PRAGMA journal_mode"))).scalar()
    assert mode == "wal"


async def test_in_db_scope_contextvar(db: Database):
    assert IN_DB_SCOPE.get() is False
    async with db.session():
        assert IN_DB_SCOPE.get() is True
    assert IN_DB_SCOPE.get() is False


async def test_app_db_unbound_raises():
    app = App("workouts", description="d")
    with pytest.raises(RuntimeNotBoundError):
        async with app.db():
            pass


async def test_dispose_delegates_to_engine(tmp_path, monkeypatch):
    """Database.dispose() is sugar over engine.dispose()."""
    d = Database(f"sqlite+aiosqlite:///{tmp_path}/dispose.db")
    calls = []
    original_dispose = type(d.engine).dispose

    async def spy_dispose(self, *args, **kwargs):
        calls.append((args, kwargs))
        return await original_dispose(self, *args, **kwargs)

    # AsyncEngine uses __slots__ (only "sync_engine") -- an instance attribute
    # named "dispose" can't be set, so the spy is patched at the class level.
    monkeypatch.setattr(type(d.engine), "dispose", spy_dispose)
    await d.dispose()
    assert len(calls) == 1

    # the pool reopens connections lazily -- the engine stays usable afterward.
    async with d.engine.begin() as conn:
        await conn.execute(text("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)"))
    async with d.session() as s:
        await s.execute(text("INSERT INTO items (name) VALUES ('after-dispose')"))
    async with d.session() as s:
        count = (await s.execute(text("SELECT COUNT(*) FROM items"))).scalar()
    assert count == 1
    await d.dispose()


async def test_app_db_bound_delegates(db: Database):
    app = App("workouts", description="d")
    app.bind_database(db)
    async with app.db() as s:
        await s.execute(text("INSERT INTO items (name) VALUES ('c')"))
    async with db.session() as s:
        count = (await s.execute(text("SELECT COUNT(*) FROM items"))).scalar()
    assert count == 1
