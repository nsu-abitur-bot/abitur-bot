"""add feedback table

Revision ID: add_feedback_table
Revises: add_message_table
Create Date: 2026-03-30
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id",
            sa.BigInteger(),
            sa.ForeignKey("user.user_id"),
            nullable=False,
        ),
        sa.Column("session_id", sa.String(100), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_feedback_user_id", "feedback", ["user_id"])
    op.create_index("idx_feedback_session_id", "feedback", ["session_id"])
    op.create_index("idx_feedback_created_at", "feedback", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_feedback_created_at", table_name="feedback")
    op.drop_index("idx_feedback_session_id", table_name="feedback")
    op.drop_index("idx_feedback_user_id", table_name="feedback")
    op.drop_table("feedback")
