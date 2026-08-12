"""Applies one first-party app's migration lane.

`version_locations` and `version_table` are set per lane by
`migrate._suite_lane_config`, so all lanes share this one env. No
target_metadata: suite revisions are hand-written and ship in the wheel, so
autogenerate never runs through here.
"""

from alembic import context
from sqlalchemy import create_engine

config = context.config


def run_migrations_online() -> None:
    engine = create_engine(config.get_main_option("sqlalchemy.url"))
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=None,
            version_table=config.get_main_option("version_table"),
            render_as_batch=engine.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run_migrations_online()
