from importlib.resources import files
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from dudamel.migrate import sync_url, upgrade_core

NEW_COLUMNS = {"actor", "source"}


def _core_cfg(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(files("dudamel") / "migrations"))
    cfg.set_main_option("sqlalchemy.url", sync_url(url))
    cfg.set_main_option("version_table", "alembic_version_core")
    return cfg


def test_fresh_upgrade_has_the_new_columns(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path}/a.db"
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    assert NEW_COLUMNS <= {c["name"] for c in insp.get_columns("activity")}


def test_upgrade_from_0005_adds_them_and_keeps_existing_rows(tmp_path: Path) -> None:
    """A pip upgrade must evolve existing installs: simulate a 0005-only DB,
    confirm it lacks the columns, and that upgrading adds them while the rows
    already in activity survive and read as unknown."""
    url = f"sqlite+aiosqlite:///{tmp_path}/b.db"
    command.upgrade(_core_cfg(url), "0005")
    engine = create_engine(sync_url(url))
    assert not (NEW_COLUMNS & {c["name"] for c in inspect(engine).get_columns("activity")})
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO conversations (id, channel, created_at) VALUES (1,'t:1',:t)"),
            {"t": "2026-01-01 00:00:00"},
        )
        conn.execute(
            text(
                "INSERT INTO activity (conversation_id, tool, args, status, result_preview, "
                "created_at) VALUES (1, 'log_workout', '{}', 'ok', 'done', :t)"
            ),
            {"t": "2026-01-01 00:00:00"},
        )

    upgrade_core(url)

    insp = inspect(create_engine(sync_url(url)))
    assert NEW_COLUMNS <= {c["name"] for c in insp.get_columns("activity")}
    with create_engine(sync_url(url)).begin() as conn:
        rows = conn.execute(text("SELECT tool, status, actor, source FROM activity")).all()
    assert rows == [("log_workout", "ok", None, None)]


def test_downgrade_from_0006_drops_the_columns_and_keeps_the_rows(tmp_path: Path) -> None:
    """`batch_alter_table` drop_column on SQLite is a full table rebuild --
    every row is copied into a new table -- so the rows surviving it is the
    part worth pinning, not just the columns going away."""
    url = f"sqlite+aiosqlite:///{tmp_path}/d.db"
    upgrade_core(url)
    engine = create_engine(sync_url(url))
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO conversations (id, channel, created_at) VALUES (1,'t:1',:t)"),
            {"t": "2026-01-01 00:00:00"},
        )
        conn.execute(
            text(
                "INSERT INTO activity (conversation_id, tool, args, status, result_preview, "
                "actor, source, created_at) VALUES (1, 'log_workout', '{}', 'ok', 'done', "
                "'session', 'web', :t)"
            ),
            {"t": "2026-01-01 00:00:00"},
        )

    command.downgrade(_core_cfg(url), "0005")

    insp = inspect(create_engine(sync_url(url)))
    assert not (NEW_COLUMNS & {c["name"] for c in insp.get_columns("activity")})
    with create_engine(sync_url(url)).begin() as conn:
        rows = conn.execute(text("SELECT conversation_id, tool, status FROM activity")).all()
    assert rows == [(1, "log_workout", "ok")]


def test_model_matches_migration(tmp_path: Path) -> None:
    """No drift between the ORM model and the migrated schema."""
    from dudamel.models_core import Activity

    url = f"sqlite+aiosqlite:///{tmp_path}/c.db"
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    assert {c["name"] for c in insp.get_columns("activity")} == set(
        Activity.__table__.columns.keys()
    )
