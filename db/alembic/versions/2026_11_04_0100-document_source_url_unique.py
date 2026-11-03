"""Keep one active document per graph and source URL."""

import sqlalchemy as sa
from alembic import op

revision = "c3a4e7b92f10"
down_revision = "8f31c9d1a204"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_document_graph_source_url_active",
        "document",
        ["graph_id", "source_url"],
        unique=True,
        postgresql_where=sa.text(
            "source_url IS NOT NULL AND status NOT IN ('deleted', 'удалён')"
        ),
    )


def downgrade() -> None:
    op.drop_index("ix_document_graph_source_url_active", table_name="document")
