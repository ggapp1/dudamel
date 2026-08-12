from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

from dudamel.exceptions import DudamelError
from dudamel.migrate import (
    pending_migrations,
    suite_version_table,
    upgrade_all,
    upgrade_core,
    upgrade_suite_app,
)

REV = '''"""{message}

Revision ID: {rev}
Revises: {down}
"""
import sqlalchemy as sa
from alembic import op

revision = {rev!r}
down_revision = {down!r}
branch_labels = None
depends_on = None


def upgrade() -> None:
    {body}


def downgrade() -> None:
    raise NotImplementedError("app migrations are forward-only in dudamel v1")
'''


def make_lane(tmp_path: Path, app: str, *, body: str, rev: str, down: str | None = None) -> Path:
    versions = tmp_path / f"{app}_versions"
    versions.mkdir(exist_ok=True)
    (versions / f"{rev}.py").write_text(
        REV.format(message=f"{app} {rev}", rev=rev, down=down, body=body)
    )
    return versions


def db_url_for(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{tmp_path / 'x.db'}"


def table_names(db_url: str) -> set[str]:
    engine = create_engine(db_url.replace("+aiosqlite", ""))
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


CREATE_NOTES = "op.create_table('notes_note', sa.Column('id', sa.Integer(), primary_key=True))"
CREATE_TASKS = "op.create_table('tasks_task', sa.Column('id', sa.Integer(), primary_key=True))"


def test_lane_creates_tables_in_its_own_version_table(tmp_path) -> None:
    url = db_url_for(tmp_path)
    upgrade_core(url)
    upgrade_suite_app(url, "notes", make_lane(tmp_path, "notes", body=CREATE_NOTES, rev="n1"))
    names = table_names(url)
    assert "notes_note" in names
    assert suite_version_table("notes") in names


def test_lanes_are_independent(tmp_path) -> None:
    url = db_url_for(tmp_path)
    upgrade_core(url)
    lanes = [
        ("notes", make_lane(tmp_path, "notes", body=CREATE_NOTES, rev="n1")),
        ("tasks", make_lane(tmp_path, "tasks", body=CREATE_TASKS, rev="t1")),
    ]
    upgrade_all(url, tmp_path, lanes)
    engine = create_engine(url.replace("+aiosqlite", ""))
    try:
        with engine.connect() as conn:
            notes_rev = conn.execute(
                text(f"select version_num from {suite_version_table('notes')}")
            ).scalar()
            tasks_rev = conn.execute(
                text(f"select version_num from {suite_version_table('tasks')}")
            ).scalar()
    finally:
        engine.dispose()
    assert notes_rev == "n1"
    assert tasks_rev == "t1"


def test_disabling_a_lane_leaves_its_data(tmp_path) -> None:
    url = db_url_for(tmp_path)
    upgrade_core(url)
    notes = make_lane(tmp_path, "notes", body=CREATE_NOTES, rev="n1")
    upgrade_all(url, tmp_path, [("notes", notes)])
    engine = create_engine(url.replace("+aiosqlite", ""))
    with engine.begin() as conn:
        conn.execute(text("insert into notes_note (id) values (1)"))
    engine.dispose()

    upgrade_all(url, tmp_path, [])  # notes now disabled

    assert "notes_note" in table_names(url)
    engine = create_engine(url.replace("+aiosqlite", ""))
    try:
        with engine.connect() as conn:
            assert conn.execute(text("select count(*) from notes_note")).scalar() == 1
    finally:
        engine.dispose()

    upgrade_all(url, tmp_path, [("notes", notes)])  # re-enabled, data intact
    engine = create_engine(url.replace("+aiosqlite", ""))
    try:
        with engine.connect() as conn:
            assert conn.execute(text("select count(*) from notes_note")).scalar() == 1
    finally:
        engine.dispose()


def test_failing_lane_stops_later_lanes_and_reruns_cleanly(tmp_path) -> None:
    """Sequential, not atomic: earlier lanes stay applied, later ones do not
    run, and a rerun after the fix completes without manual repair."""
    url = db_url_for(tmp_path)
    upgrade_core(url)
    a = make_lane(tmp_path, "aaa", body=CREATE_NOTES.replace("notes_note", "aaa_t"), rev="a1")
    bad = make_lane(tmp_path, "bbb", body="raise RuntimeError('boom')", rev="b1")
    z = make_lane(tmp_path, "zzz", body=CREATE_TASKS.replace("tasks_task", "zzz_t"), rev="z1")

    with pytest.raises(DudamelError, match="bbb"):
        upgrade_all(url, tmp_path, [("aaa", a), ("bbb", bad), ("zzz", z)])

    names = table_names(url)
    assert "aaa_t" in names, "an already-applied lane must stay applied"
    assert "zzz_t" not in names, "a later lane must not run after a failure"

    fixed = make_lane(tmp_path, "bbb", body=CREATE_NOTES.replace("notes_note", "bbb_t"), rev="b1")
    upgrade_all(url, tmp_path, [("aaa", a), ("bbb", fixed), ("zzz", z)])
    names = table_names(url)
    assert {"aaa_t", "bbb_t", "zzz_t"} <= names


def test_lane_versions_dir_may_contain_spaces_and_commas(tmp_path) -> None:
    """Alembic's legacy version_locations splitting would silently find no
    revisions under a path containing a space or a comma."""
    url = db_url_for(tmp_path)
    upgrade_core(url)
    awkward = tmp_path / "My Apps, v2"
    awkward.mkdir()
    notes = make_lane(awkward, "notes", body=CREATE_NOTES, rev="n1")
    assert pending_migrations(url, tmp_path, [("notes", notes)]) == [
        "app 'notes' schema is behind head"
    ]
    upgrade_all(url, tmp_path, [("notes", notes)])
    assert "notes_note" in table_names(url)


def test_upgrade_all_without_a_project_lane(tmp_path) -> None:
    """A project that has never generated a revision has no migrations/
    directory; the suite lanes must still apply."""
    url = db_url_for(tmp_path)
    upgrade_core(url)
    assert not (tmp_path / "migrations").exists()
    upgrade_all(
        url, tmp_path, [("notes", make_lane(tmp_path, "notes", body=CREATE_NOTES, rev="n1"))]
    )
    assert "notes_note" in table_names(url)
    assert not (tmp_path / "migrations").exists()


def test_app_with_no_revisions_is_a_harmless_no_op(tmp_path) -> None:
    """An app that ships no versions directory (no tables of its own) must
    neither fail its lane nor be reported as pending."""
    url = db_url_for(tmp_path)
    upgrade_core(url)
    empty = tmp_path / "noschema" / "versions"
    upgrade_all(url, tmp_path, [("noschema", empty)])
    assert pending_migrations(url, tmp_path, [("noschema", empty)]) == []
    upgrade_all(url, tmp_path, [("noschema", empty)])  # still a no-op on rerun
    assert pending_migrations(url, tmp_path, [("noschema", empty)]) == []


def test_pending_migrations_names_each_lane(tmp_path) -> None:
    url = db_url_for(tmp_path)
    upgrade_core(url)
    notes = make_lane(tmp_path, "notes", body=CREATE_NOTES, rev="n1")
    pending = pending_migrations(url, tmp_path, [("notes", notes)])
    assert any("notes" in p for p in pending)
    upgrade_all(url, tmp_path, [("notes", notes)])
    assert pending_migrations(url, tmp_path, [("notes", notes)]) == []
