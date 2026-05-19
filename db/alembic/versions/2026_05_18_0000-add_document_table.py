"""Add document table

Revision ID: 4b5c6d7e8f90
Revises: 3a4b5c6d7e8f
Create Date: 2026-05-18 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "4b5c6d7e8f90"
down_revision: Union[str, None] = "3a4b5c6d7e8f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "document",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("source_url", sa.String(length=1000), nullable=True),
        sa.Column("graph_id", sa.String(length=100), nullable=False),
        sa.Column("rag_doc_id", sa.String(length=500), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("content_length", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("last_indexed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "graph_id", "rag_doc_id", name="uq_document_graph_rag_doc_id"
        ),
    )
    op.create_index("ix_document_graph_id", "document", ["graph_id"])
    op.create_index("ix_document_source_url", "document", ["source_url"])
    op.create_index("ix_document_content_hash", "document", ["content_hash"])
    op.create_index("ix_document_status", "document", ["status"])


def downgrade() -> None:
    op.drop_index("ix_document_status", table_name="document")
    op.drop_index("ix_document_content_hash", table_name="document")
    op.drop_index("ix_document_source_url", table_name="document")
    op.drop_index("ix_document_graph_id", table_name="document")
    op.drop_table("document")
