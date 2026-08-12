from importlib.resources import files
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from dudamel.migrate import sync_url, upgrade_core

DEAD_COLUMNS = {"tokens_in", "tokens_out", "cost_usd"}
REDUNDANT_INDEX = "ix_summaries_conversation_id"
KEPT_INDEX = "ix_summaries_conversation_id_id"


def _core_cfg(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(files("dudamel") / "migrations"))
    cfg.set_main_option("sqlalchemy.url", sync_url(url))
    cfg.set_main_option("version_table", "alembic_version_core")
    return cfg


def test_fresh_upgrade_has_neither_the_dead_columns_nor_the_index(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path}/a.db"
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    activity_columns = {c["name"] for c in insp.get_columns("activity")}
    assert not (DEAD_COLUMNS & activity_columns)
    index_names = {ix["name"] for ix in insp.get_indexes("summaries")}
    assert REDUNDANT_INDEX not in index_names
    assert KEPT_INDEX in index_names  # the composite one still serves the lookups


def test_upgrade_from_0004_drops_them(tmp_path: Path) -> None:
    """A pip upgrade must evolve existing 0004 installs: simulate a
    0004-only DB, confirm it still carries both, and that upgrading removes
    them without disturbing the rows already in activity."""
    url = f"sqlite+aiosqlite:///{tmp_path}/b.db"
    command.upgrade(_core_cfg(url), "0004")
    engine = create_engine(sync_url(url))
    insp = inspect(engine)
    assert DEAD_COLUMNS <= {c["name"] for c in insp.get_columns("activity")}
    assert REDUNDANT_INDEX in {ix["name"] for ix in insp.get_indexes("summaries")}
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO conversations (id, channel, created_at) VALUES (1,'t:1',:t)"),
            {"t": "2026-01-01 00:00:00"},
        )
        conn.execute(
            text(
                "INSERT INTO activity (conversation_id, tool, args, status, result_preview, "
                "tokens_in, created_at) VALUES (1, 'log_workout', '{}', 'ok', 'done', 7, :t)"
            ),
            {"t": "2026-01-01 00:00:00"},
        )

    upgrade_core(url)

    insp = inspect(create_engine(sync_url(url)))
    assert not (DEAD_COLUMNS & {c["name"] for c in insp.get_columns("activity")})
    assert REDUNDANT_INDEX not in {ix["name"] for ix in insp.get_indexes("summaries")}
    with create_engine(sync_url(url)).begin() as conn:
        row = conn.execute(
            text("SELECT conversation_id, tool, status, result_preview FROM activity")
        ).all()
    assert row == [(1, "log_workout", "ok", "done")]  # the surviving data is intact


def test_model_matches_migration(tmp_path: Path) -> None:
    """No drift between the ORM models and the migrated schema."""
    from dudamel.models_core import Activity, Summary

    url = f"sqlite+aiosqlite:///{tmp_path}/c.db"
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    assert {c["name"] for c in insp.get_columns("activity")} == set(
        Activity.__table__.columns.keys()
    )
    migrated = {ix["name"] for ix in insp.get_indexes("summaries")}
    assert migrated == {ix.name for ix in Summary.__table__.indexes}
