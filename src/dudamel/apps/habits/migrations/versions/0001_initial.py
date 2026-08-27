"""create the habits app's tables

Columns read out of `app.metadata`, not written from memory.

The unique constraint on (habit_id, day) is load-bearing, not tidiness: the tick
button is a one-tap POST behind a browser guard the suite cannot execute, and
the model can call the tool twice in one batch. It is what makes a duplicate a
no-op rather than a second row.

`day` is a LOCAL date, unlike every datetime in this schema, which is naive-UTC.
The day a tick belongs to is the user's day.

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
        "habits_habits",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "habits_ticks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("habit_id", sa.Integer(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.UniqueConstraint("habit_id", "day", name="uq_habits_ticks_habit_day"),
    )


def downgrade() -> None:
    op.drop_table("habits_ticks")
    op.drop_table("habits_habits")
