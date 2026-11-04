"""add answer feedback candidates

Revision ID: 8f31c9d1a204
Revises: a6b7c8d9e0f1
"""

import sqlalchemy as sa
from alembic import op

revision = "8f31c9d1a204"
down_revision = "a6b7c8d9e0f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "answer_feedback",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("session_id", sa.String(length=255), nullable=False),
        sa.Column("channel", sa.String(length=50), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("bot_response", sa.Text(), nullable=False),
        sa.Column("rating", sa.String(length=16), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("rated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_answer_feedback_rating", "answer_feedback", ["rating"])
    op.create_index("ix_answer_feedback_created_at", "answer_feedback", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_answer_feedback_created_at", table_name="answer_feedback")
    op.drop_index("ix_answer_feedback_rating", table_name="answer_feedback")
    op.drop_table("answer_feedback")
