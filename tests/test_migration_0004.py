from importlib.resources import files
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from dudamel.migrate import sync_url, upgrade_core

TABLE = "summaries"
UNIQUE_INDEX = "uq_summaries_conv_upto"
COMPOSITE_INDEX = "ix_summaries_conversation_id_id"


def _core_cfg(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(files("dudamel") / "migrations"))
    cfg.set_main_option("sqlalchemy.url", sync_url(url))
    cfg.set_main_option("version_table", "alembic_version_core")
    return cfg


def test_fresh_upgrade_creates_summaries_table(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path}/a.db"
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    assert TABLE in insp.get_table_names()
    names = {ix["name"] for ix in insp.get_indexes(TABLE)}
    assert UNIQUE_INDEX in names
    assert COMPOSITE_INDEX in names


def test_upgrade_from_0003_to_head(tmp_path: Path) -> None:
    """A pip upgrade must evolve existing 0003 installs: simulate a
    0003-only DB and confirm it gains the summaries table on upgrade."""
    url = f"sqlite+aiosqlite:///{tmp_path}/b.db"
    command.upgrade(_core_cfg(url), "0003")
    insp = inspect(create_engine(sync_url(url)))
    assert TABLE not in insp.get_table_names()
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    assert TABLE in insp.get_table_names()


def test_unique_index_is_unique_on_the_right_columns(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path}/c.db"
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    ix = next(ix for ix in insp.get_indexes(TABLE) if ix["name"] == UNIQUE_INDEX)
    assert ix["unique"]
    assert ix["column_names"] == ["conversation_id", "up_to_message_id"]


def test_model_matches_migration_columns(tmp_path: Path) -> None:
    """No drift between the ORM model's columns and the migration DDL:
    same column names, same nullability."""
    from dudamel.models_core import Summary

    url = f"sqlite+aiosqlite:///{tmp_path}/d.db"
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    db_columns = {c["name"]: c["nullable"] for c in insp.get_columns(TABLE)}
    model_columns = {c.name: c.nullable for c in Summary.__table__.columns}
    assert db_columns == model_columns


def test_model_matches_migration_indexes(tmp_path: Path) -> None:
    from dudamel.models_core import Summary

    url = f"sqlite+aiosqlite:///{tmp_path}/e.db"
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    migration_index = next(ix for ix in insp.get_indexes(TABLE) if ix["name"] == UNIQUE_INDEX)
    model_index = next(ix for ix in Summary.__table__.indexes if ix.name == UNIQUE_INDEX)
    assert model_index.unique
    assert [c.name for c in model_index.columns] == migration_index["column_names"]


def test_downgrade_from_0004_drops_table(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path}/f.db"
    upgrade_core(url)
    command.downgrade(_core_cfg(url), "0003")
    insp = inspect(create_engine(sync_url(url)))
    assert TABLE not in insp.get_table_names()
