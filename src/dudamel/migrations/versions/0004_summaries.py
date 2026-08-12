"""summaries table

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "summaries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "conversation_id", sa.Integer(), sa.ForeignKey("conversations.id"), nullable=False
        ),
        sa.Column("up_to_message_id", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("tainted", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_summaries_conversation_id", "summaries", ["conversation_id"])
    op.create_index("ix_summaries_conversation_id_id", "summaries", ["conversation_id", "id"])
    op.create_index(
        "uq_summaries_conv_upto",
        "summaries",
        ["conversation_id", "up_to_message_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("summaries")
