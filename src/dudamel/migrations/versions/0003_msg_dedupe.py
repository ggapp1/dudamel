"""messages dedupe unique index

Revision ID: 0003
Revises: 0002
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "uq_messages_conv_client_msg",
        "messages",
        ["conversation_id", "client_msg_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_messages_conv_client_msg", table_name="messages")
