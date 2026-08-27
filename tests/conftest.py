"""Shared fixtures for the first-party app suite.

`App` instances are module-level singletons shared by every test in the session,
so a test that binds a database or settings and does not restore them leaks into
every later test in collection order. These fixtures snapshot and restore.

Binding settings is not optional: `App.settings` raises RuntimeNotBoundError
when unbound (app.py), and `run_widget` swallows that into an error card with no
"data" key -- so an unbound app surfaces as a bare KeyError far from its cause.
"""

from __future__ import annotations

import contextlib
import importlib
from typing import Any

import pytest

from dudamel.db import Database


@contextlib.asynccontextmanager
async def bound_app(module_name: str, tmp_path: Any, settings: dict | None = None):
    app = importlib.import_module(module_name).app
    previous_db, previous_settings = app._database, app._settings
    db = Database(f"sqlite+aiosqlite:///{tmp_path}/{app.name}.db")
    try:
        async with db.engine.begin() as conn:
            await conn.run_sync(app.metadata.create_all)
        app.bind_database(db)
        app.bind_settings(settings or {})
        yield app
    finally:
        await db.dispose()
        # Reaching into the private attributes is deliberate: there is no public
        # restore seam, and leaving a disposed Database bound is worse.
        app._database, app._settings = previous_db, previous_settings


@pytest.fixture
async def tasks_app(tmp_path):
    async with bound_app("dudamel.apps.tasks", tmp_path) as app:
        yield app


@pytest.fixture
async def notes_app(tmp_path):
    async with bound_app("dudamel.apps.notes", tmp_path) as app:
        yield app


@pytest.fixture
async def habits_app(tmp_path):
    async with bound_app("dudamel.apps.habits", tmp_path) as app:
        yield app
