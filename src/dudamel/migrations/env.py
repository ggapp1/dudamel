from alembic import context
from sqlalchemy import create_engine

from dudamel.models_core import CoreBase

config = context.config
target_metadata = CoreBase.metadata
VERSION_TABLE = config.get_main_option("version_table", "alembic_version_core")


def run_migrations_online() -> None:
    engine = create_engine(config.get_main_option("sqlalchemy.url"))
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table=VERSION_TABLE,
            render_as_batch=engine.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run_migrations_online()
