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


async def test_app_db_bound_delegates(db: Database):
    app = App("workouts", description="d")
    app.bind_database(db)
    async with app.db() as s:
        await s.execute(text("INSERT INTO items (name) VALUES ('c')"))
    async with db.session() as s:
        count = (await s.execute(text("SELECT COUNT(*) FROM items"))).scalar()
    assert count == 1
