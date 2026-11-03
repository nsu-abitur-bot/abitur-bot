"""Add verification and provenance metadata to FAQ entries."""

import sqlalchemy as sa
from alembic import op

revision = "a4c8e1f3b7d2"
down_revision = "a6b7c8d9e0f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "faq_entry",
        sa.Column(
            "verification_status",
            sa.String(length=20),
            nullable=False,
            server_default="draft",
        ),
    )
    op.add_column("faq_entry", sa.Column("source_document", sa.Text(), nullable=True))
    op.add_column("faq_entry", sa.Column("source_location", sa.Text(), nullable=True))
    op.add_column(
        "faq_entry", sa.Column("confirmed_by", sa.String(length=255), nullable=True)
    )
    op.add_column("faq_entry", sa.Column("confirmed_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("faq_entry", "confirmed_at")
    op.drop_column("faq_entry", "confirmed_by")
    op.drop_column("faq_entry", "source_location")
    op.drop_column("faq_entry", "source_document")
    op.drop_column("faq_entry", "verification_status")
