from importlib.resources import files
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from dudamel.migrate import sync_url, upgrade_core


def _core_cfg(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(files("dudamel") / "migrations"))
    cfg.set_main_option("sqlalchemy.url", sync_url(url))
    cfg.set_main_option("version_table", "alembic_version_core")
    return cfg


def test_fresh_upgrade_includes_llm_calls(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path}/a.db"
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    cols = {c["name"] for c in insp.get_columns("llm_calls")}
    assert {
        "id",
        "tier",
        "provider",
        "model",
        "tokens_in",
        "tokens_out",
        "conversation_id",
        "created_at",
    } <= cols


def test_upgrade_from_0001_to_head(tmp_path: Path) -> None:
    """A pip upgrade must evolve existing installs: simulate a 0001-only DB."""
    url = f"sqlite+aiosqlite:///{tmp_path}/b.db"
    command.upgrade(_core_cfg(url), "0001")
    insp = inspect(create_engine(sync_url(url)))
    assert "llm_calls" not in insp.get_table_names()
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    assert "llm_calls" in insp.get_table_names()


def test_model_matches_migration(tmp_path: Path) -> None:
    """No drift between the ORM model and the migration DDL."""
    from dudamel.models_core import LlmCall

    url = f"sqlite+aiosqlite:///{tmp_path}/c.db"
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    migration_cols = {c["name"] for c in insp.get_columns("llm_calls")}
    model_cols = set(LlmCall.__table__.columns.keys())
    assert migration_cols == model_cols
