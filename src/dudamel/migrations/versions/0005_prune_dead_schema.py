"""drop activity's reserved columns and the redundant summaries index

activity.tokens_in / tokens_out / cost_usd were reserved in 0001 and never
written; llm_calls (0002) became the real home of token accounting, per
call and with a conversation_id. ix_summaries_conversation_id duplicates
the leading column of ix_summaries_conversation_id_id (0004), which serves
every conversation_id lookup on both backends, so it only added write cost.

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

_DEAD_ACTIVITY_COLUMNS = ("tokens_in", "tokens_out", "cost_usd")


def upgrade() -> None:
    # Batch mode so SQLite gets a table rebuild rather than relying on a
    # particular ALTER TABLE ... DROP COLUMN version (env.py already sets
    # render_as_batch for autogenerate; this states it for the operation).
    with op.batch_alter_table("activity") as batch:
        for name in _DEAD_ACTIVITY_COLUMNS:
            batch.drop_column(name)
    op.drop_index("ix_summaries_conversation_id", table_name="summaries")


def downgrade() -> None:
    op.create_index("ix_summaries_conversation_id", "summaries", ["conversation_id"])
    with op.batch_alter_table("activity") as batch:
        batch.add_column(sa.Column("tokens_in", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("tokens_out", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("cost_usd", sa.Float(), nullable=True))
