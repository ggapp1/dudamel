"""record who invoked an activity row and through which surface

Until now every activity row came from the router on the model's behalf, so
neither question had an answer worth storing. The deterministic plane adds a
second, human actor on two different surfaces, and an audit log that cannot
say which one acted is not an audit log.

Both columns are nullable: rows written before this migration cannot be
attributed, and backfilling them with a guess would put fiction in the log.

Revision ID: 0006
Revises: 0005
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Batch mode so SQLite gets a table rebuild rather than relying on a
    # particular ALTER TABLE version (env.py already sets render_as_batch for
    # autogenerate; this states it for the operation).
    with op.batch_alter_table("activity") as batch:
        batch.add_column(sa.Column("actor", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("source", sa.String(length=16), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("activity") as batch:
        batch.drop_column("source")
        batch.drop_column("actor")
