"""Add Mini App conversations and turns."""

import sqlalchemy as sa
from alembic import op

revision = "fac64d37b12e"
down_revision = "0af365dca483"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "mini_app_conversation",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("instance_id", sa.String(64), nullable=False),
        sa.Column("platform", sa.String(8), nullable=False),
        sa.Column("external_id", sa.BigInteger(), nullable=False),
        sa.Column("title", sa.String(80), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_mini_app_conversation_owner",
        "mini_app_conversation",
        ["instance_id", "platform", "external_id", "updated_at"],
    )
    op.create_table(
        "mini_app_turn",
        sa.Column(
            "conversation_id",
            sa.String(36),
            sa.ForeignKey("mini_app_conversation.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(8), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("sources", sa.JSON(), nullable=False),
        sa.Column("history_entry", sa.JSON(), nullable=False),
        sa.Column(
            "feedback_id",
            sa.String(36),
            sa.ForeignKey("answer_feedback.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_mini_app_turn_conversation_order",
        "mini_app_turn",
        ["conversation_id", "started_at", "id"],
    )


def downgrade() -> None:
    op.drop_table("mini_app_turn")
    op.drop_table("mini_app_conversation")
