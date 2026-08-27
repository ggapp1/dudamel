"""create the tasks app's table

Column definitions are taken from `app.metadata`, not written from memory: the
models are what the app code uses and this lane is what production runs, so any
drift between them is invisible until someone's database is wrong. Defaults are
Python-side (`modelsugar` supplies them at insert time), so no server_default
appears here.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tasks_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("done", sa.Boolean(), nullable=False),
        sa.Column("due", sa.Date(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("tasks_items")
