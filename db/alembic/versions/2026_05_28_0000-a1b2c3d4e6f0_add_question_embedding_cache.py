"""add question embedding cache

Revision ID: a1b2c3d4e6f0
Revises: e2f3a4b5c6d7
Create Date: 2026-05-28 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e6f0"
down_revision: Union[str, Sequence[str], None] = "e2f3a4b5c6d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "question_embedding_cache",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("question_hash", sa.String(length=64), nullable=False),
        sa.Column("canonical_question", sa.Text(), nullable=False),
        sa.Column("embedding", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider",
            "model",
            "question_hash",
            name="uq_question_embedding_cache_provider_model_hash",
        ),
    )
    op.create_index(
        "ix_question_embedding_cache_lookup",
        "question_embedding_cache",
        ["provider", "model", "question_hash"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_question_embedding_cache_lookup",
        table_name="question_embedding_cache",
    )
    op.drop_table("question_embedding_cache")
