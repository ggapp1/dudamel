from importlib.resources import files
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from dudamel.migrate import sync_url, upgrade_core

INDEX_NAME = "uq_messages_conv_client_msg"


def _core_cfg(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(files("dudamel") / "migrations"))
    cfg.set_main_option("sqlalchemy.url", sync_url(url))
    cfg.set_main_option("version_table", "alembic_version_core")
    return cfg


def test_fresh_upgrade_includes_dedupe_index(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path}/a.db"
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    names = {ix["name"] for ix in insp.get_indexes("messages")}
    assert INDEX_NAME in names


def test_upgrade_from_0002_to_head(tmp_path: Path) -> None:
    """A pip upgrade must evolve existing 0002 installs: simulate a
    0002-only DB and confirm it gains the unique dedupe index on upgrade."""
    url = f"sqlite+aiosqlite:///{tmp_path}/b.db"
    command.upgrade(_core_cfg(url), "0002")
    insp = inspect(create_engine(sync_url(url)))
    assert INDEX_NAME not in {ix["name"] for ix in insp.get_indexes("messages")}
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    assert INDEX_NAME in {ix["name"] for ix in insp.get_indexes("messages")}


def test_dedupe_index_is_unique_on_the_right_columns(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path}/c.db"
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    ix = next(ix for ix in insp.get_indexes("messages") if ix["name"] == INDEX_NAME)
    assert ix["unique"]
    assert ix["column_names"] == ["conversation_id", "client_msg_id"]


def test_model_matches_migration(tmp_path: Path) -> None:
    """No drift between the ORM model's __table_args__ unique index and the
    migration DDL."""
    from dudamel.models_core import Message

    url = f"sqlite+aiosqlite:///{tmp_path}/d.db"
    upgrade_core(url)
    insp = inspect(create_engine(sync_url(url)))
    migration_index = next(
        (ix for ix in insp.get_indexes("messages") if ix["name"] == INDEX_NAME), None
    )
    assert migration_index is not None
    assert migration_index["unique"]

    model_index = next(ix for ix in Message.__table__.indexes if ix.name == INDEX_NAME)
    assert model_index.unique
    assert [c.name for c in model_index.columns] == migration_index["column_names"]


def test_downgrade_from_0003_drops_index(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path}/e.db"
    upgrade_core(url)
    command.downgrade(_core_cfg(url), "0002")
    insp = inspect(create_engine(sync_url(url)))
    assert INDEX_NAME not in {ix["name"] for ix in insp.get_indexes("messages")}
