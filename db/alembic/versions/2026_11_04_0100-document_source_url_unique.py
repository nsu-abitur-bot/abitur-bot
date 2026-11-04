"""Keep one active document per graph and source URL."""

import sqlalchemy as sa
from alembic import op

revision = "c3a4e7b92f10"
down_revision = "8f31c9d1a204"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Keep the same winner as get_by_source_url. Soft deletion preserves all
    # metadata and RAG identities; no indexed content is physically removed.
    # Lock out writes until the index exists so no new duplicate can slip in.
    op.execute("LOCK TABLE document IN SHARE ROW EXCLUSIVE MODE")
    op.execute(
        sa.text("""
            WITH ranked AS (
                SELECT id, row_number() OVER (
                    PARTITION BY graph_id, source_url
                    ORDER BY updated_at DESC, id DESC
                ) AS position
                FROM document
                WHERE source_url IS NOT NULL AND source_url <> ''
                  AND status NOT IN ('deleted', 'удалён')
            )
            UPDATE document SET status = 'удалён'
            FROM ranked
            WHERE document.id = ranked.id AND ranked.position > 1
        """)
    )
    op.create_index(
        "ix_document_graph_source_url_active",
        "document",
        ["graph_id", "source_url"],
        unique=True,
        postgresql_where=sa.text(
            "source_url IS NOT NULL AND source_url <> '' "
            "AND status NOT IN ('deleted', 'удалён')"
        ),
    )


def downgrade() -> None:
    # Do not resurrect reconciled duplicates: newer writes may reuse their URL.
    op.drop_index("ix_document_graph_source_url_active", table_name="document")
