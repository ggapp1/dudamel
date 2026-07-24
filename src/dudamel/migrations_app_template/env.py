"""Applies app migrations. No target_metadata: autogenerate happens
programmatically via `dudamel db migrate`, never through this env."""

from alembic import context
from sqlalchemy import create_engine

config = context.config


def run_migrations_online() -> None:
    engine = create_engine(config.get_main_option("sqlalchemy.url"))
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=None,
            version_table="alembic_version_apps",
            render_as_batch=engine.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run_migrations_online()
