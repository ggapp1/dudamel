from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from dudamel import App
from dudamel.db import Database
from dudamel.exceptions import RegistryError


def make_app() -> App:
    return App("workouts", description="d")


def test_table_naming_and_prefix():
    app = make_app()

    class WorkoutSet(app.Model, table="sets"):
        exercise: str

    class Session(app.Model):
        note: str

    assert WorkoutSet.__tablename__ == "workouts_sets"
    assert Session.__tablename__ == "workouts_session"
    assert "workouts_sets" in app.metadata.tables


def test_auto_id_primary_key():
    app = make_app()

    class Thing(app.Model):
        name: str

    assert Thing.__table__.primary_key.columns.keys() == ["id"]


def test_optional_becomes_nullable():
    app = make_app()

    class Note(app.Model):
        body: str
        tag: str | None

    t = app.metadata.tables["workouts_note"]
    assert t.columns["body"].nullable is False
    assert t.columns["tag"].nullable is True


def test_unsupported_type_rejected():
    app = make_app()
    with pytest.raises(RegistryError, match="unsupported column type"):

        class Bad(app.Model):
            x: complex


async def test_roundtrip_insert_and_now_default(tmp_path):
    app = make_app()

    class WorkoutSet(app.Model, table="sets"):
        exercise: str
        reps: int = 5
        logged_at: datetime = app.now()

    db = Database(f"sqlite+aiosqlite:///{tmp_path}/m.db")
    async with db.engine.begin() as conn:
        await conn.run_sync(app.metadata.create_all)
    app.bind_database(db)

    before = datetime.now(UTC).replace(tzinfo=None)
    async with app.db() as s:
        s.add(WorkoutSet(exercise="bench"))
    async with app.db() as s:
        row = (await s.execute(select(WorkoutSet))).scalar_one()
    assert row.exercise == "bench" and row.reps == 5
    assert row.logged_at is not None and row.logged_at >= before
    await db.engine.dispose()
