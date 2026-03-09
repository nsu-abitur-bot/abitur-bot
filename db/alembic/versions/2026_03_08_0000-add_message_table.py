"""add message table

Revision ID: add_message_table
Revises: 2026_03_02_0004-add_index_on_created_at
Create Date: 2026-03-08
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "message",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id",
            sa.BigInteger(),
            sa.ForeignKey("user.user_id"),
            nullable=False,
        ),
        sa.Column("session_id", sa.String(100), nullable=False),
        sa.Column("user_text", sa.Text(), nullable=False),
        sa.Column("bot_response", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_message_user_id", "message", ["user_id"])
    op.create_index("idx_message_session_id", "message", ["session_id"])
    op.create_index("idx_message_created_at", "message", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_message_created_at", table_name="message")
    op.drop_index("idx_message_session_id", table_name="message")
    op.drop_index("idx_message_user_id", table_name="message")
    op.drop_table("message")
