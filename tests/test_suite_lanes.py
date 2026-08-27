"""Every shipped app's migration lane, applied for real.

The app test files build their schema with `metadata.create_all` -- from the
MODELS. The lane is what runs on a user's machine. Nothing else in this suite
executes one, so without these tests a revision that creates a mistyped table,
omits a column, or breaks its chain passes everything and is discovered only
when somebody's database is wrong.
"""

import importlib

import pytest
from sqlalchemy import create_engine, inspect

from dudamel.apps import SUITE_APPS, suite_versions_dir
from dudamel.migrate import upgrade_core, upgrade_suite_app

ENTRIES = sorted(SUITE_APPS.values(), key=lambda entry: entry.name)
# Parametrising over an empty collection SKIPS by default rather than failing,
# so an empty registry would read green. Fail at collection instead.
assert ENTRIES, "SUITE_APPS is empty; every parametrised test below would skip"


def _url(tmp_path, name: str) -> str:
    return f"sqlite+aiosqlite:///{tmp_path / f'{name}.db'}"


def _inspector(db_url: str):
    engine = create_engine(db_url.replace("+aiosqlite", ""))
    try:
        return inspect(engine), engine
    except Exception:
        engine.dispose()
        raise


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
def test_lane_applies_and_matches_its_models(tmp_path, entry):
    """The lane and the models must agree. Drift between them is invisible to
    every other test in the suite."""
    db_url = _url(tmp_path, entry.name)
    upgrade_core(db_url)
    upgrade_suite_app(db_url, entry.name, suite_versions_dir(entry))

    app = importlib.import_module(entry.module).app
    inspector, engine = _inspector(db_url)
    try:
        created = set(inspector.get_table_names())
        for table_name, table in app.metadata.tables.items():
            assert table_name in created, f"{entry.name}: the lane never created {table_name}"
            assert {c["name"] for c in inspector.get_columns(table_name)} == set(
                table.columns.keys()
            ), f"{entry.name}: {table_name} columns drifted from the model"
    finally:
        engine.dispose()


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
def test_lane_is_idempotent(tmp_path, entry):
    """`db migrate` runs on every start; a second run must be a no-op."""
    db_url = _url(tmp_path, entry.name)
    upgrade_core(db_url)
    upgrade_suite_app(db_url, entry.name, suite_versions_dir(entry))
    upgrade_suite_app(db_url, entry.name, suite_versions_dir(entry))


def test_enabling_one_app_creates_only_its_own_tables(tmp_path):
    """One app's migration problem is exactly one app's problem."""
    db_url = _url(tmp_path, "only-notes")
    upgrade_core(db_url)
    upgrade_suite_app(db_url, "notes", suite_versions_dir(SUITE_APPS["notes"]))

    inspector, engine = _inspector(db_url)
    try:
        tables = set(inspector.get_table_names())
    finally:
        engine.dispose()

    assert "notes_entries" in tables
    assert not [t for t in tables if t.startswith(("tasks_", "habits_"))]


def test_the_habits_tick_constraint_survives_into_the_migrated_schema(tmp_path):
    """The constraint lives in `__table_args__`, so `create_all` would create it
    even if the revision forgot. Only the applied lane can prove it ships."""
    db_url = _url(tmp_path, "habits-constraint")
    upgrade_core(db_url)
    upgrade_suite_app(db_url, "habits", suite_versions_dir(SUITE_APPS["habits"]))

    inspector, engine = _inspector(db_url)
    try:
        constraints = inspector.get_unique_constraints("habits_ticks")
    finally:
        engine.dispose()

    assert [set(c["column_names"]) for c in constraints] == [{"habit_id", "day"}]
