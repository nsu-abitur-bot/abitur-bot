"""add feedback report table

Revision ID: 9f1a2b3c4d5e
Revises: d267a76e1a85
Create Date: 2026-05-26 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9f1a2b3c4d5e"
down_revision: Union[str, Sequence[str], None] = "d267a76e1a85"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "feedback_report",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("session_id", sa.String(length=255), nullable=False),
        sa.Column("channel", sa.String(length=50), nullable=False),
        sa.Column("comment", sa.Text(), nullable=False),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("bot_response", sa.Text(), nullable=True),
        sa.Column("logs_snapshot", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_feedback_report_user_id", "feedback_report", ["user_id"], unique=False
    )
    op.create_index(
        "ix_feedback_report_session_id",
        "feedback_report",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        "ix_feedback_report_status", "feedback_report", ["status"], unique=False
    )
    op.create_index(
        "ix_feedback_report_created_at",
        "feedback_report",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_feedback_report_created_at", table_name="feedback_report")
    op.drop_index("ix_feedback_report_status", table_name="feedback_report")
    op.drop_index("ix_feedback_report_session_id", table_name="feedback_report")
    op.drop_index("ix_feedback_report_user_id", table_name="feedback_report")
    op.drop_table("feedback_report")
