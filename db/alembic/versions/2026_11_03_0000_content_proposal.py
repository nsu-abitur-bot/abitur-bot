"""Add review queue for generated and parsed content."""

import sqlalchemy as sa
from alembic import op

revision = "a6b7c8d9e0f1"
down_revision = "7b4e1d9c2a60"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "content_proposal",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("graph_id", sa.String(100), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("source_url", sa.String(1000), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("reviewer_id", sa.String(36), nullable=True),
        sa.Column("document_id", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_content_proposal_status_created",
        "content_proposal",
        ["status", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_content_proposal_status_created", table_name="content_proposal")
    op.drop_table("content_proposal")
