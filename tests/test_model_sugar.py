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


def test_abstract_mixin_annotations_are_not_dropped():
    app = make_app()

    class Timestamped(app.Model):
        __abstract__ = True
        created_at: datetime = app.now()

    class WorkoutSet(Timestamped):
        name: str

    # Both the mixin's field and the subclass's own field become columns.
    cols = WorkoutSet.__table__.columns.keys()
    assert "created_at" in cols
    assert "name" in cols
    assert "id" in cols


def test_abstract_mixin_subclass_override_wins():
    app = make_app()

    class Timestamped(app.Model):
        __abstract__ = True
        created_at: datetime = app.now()

    fixed = datetime(2020, 1, 1, 12, 0)

    class Custom(Timestamped):
        created_at: datetime = fixed  # overrides the mixin's app.now() default
        label: str

    # The mixin default (a callable factory) must NOT leak through — the
    # subclass's plain scalar default wins.
    assert Custom.__table__.columns["created_at"].default.arg == fixed


async def test_abstract_mixin_roundtrip_populates_mixin_field(tmp_path):
    app = make_app()

    class Timestamped(app.Model):
        __abstract__ = True
        created_at: datetime = app.now()

    class WorkoutSet(Timestamped):
        name: str

    db = Database(f"sqlite+aiosqlite:///{tmp_path}/mixin.db")
    async with db.engine.begin() as conn:
        await conn.run_sync(app.metadata.create_all)
    app.bind_database(db)

    before = datetime.now(UTC).replace(tzinfo=None)
    async with app.db() as s:
        s.add(WorkoutSet(name="thing"))
    async with app.db() as s:
        row = (await s.execute(select(WorkoutSet))).scalar_one()
    assert row.name == "thing"
    assert row.created_at is not None and row.created_at >= before
    await db.engine.dispose()


def test_subclassing_concrete_model_raises():
    app = make_app()

    class A(app.Model):
        name: str

    with pytest.raises(RegistryError, match="abstract"):

        class B(A):
            extra: str


def test_bare_str_id_becomes_primary_key_no_auto_id():
    app = make_app()

    class Ticket(app.Model):
        id: str
        label: str

    assert Ticket.__table__.primary_key.columns.keys() == ["id"]
    assert sorted(Ticket.__table__.columns.keys()) == ["id", "label"]
    assert Ticket.__table__.columns["id"].autoincrement is False


async def test_bare_str_id_insert_roundtrip(tmp_path):
    app = make_app()

    class Ticket(app.Model):
        id: str
        label: str

    db = Database(f"sqlite+aiosqlite:///{tmp_path}/ticket.db")
    async with db.engine.begin() as conn:
        await conn.run_sync(app.metadata.create_all)
    app.bind_database(db)

    async with app.db() as s:
        s.add(Ticket(id="abc123", label="hi"))
    async with app.db() as s:
        row = (await s.execute(select(Ticket))).scalar_one()
    assert row.id == "abc123" and row.label == "hi"
    await db.engine.dispose()


def test_id_with_default_rejected():
    app = make_app()
    with pytest.raises(RegistryError, match="defaults are not supported"):

        class Bad(app.Model):
            id: int = 5
