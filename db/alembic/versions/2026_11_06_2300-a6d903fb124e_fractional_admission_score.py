"""Preserve fractional college admission scores.

Revision ID: a6d903fb124e
Revises: 9c7b8608e613
"""

import sqlalchemy as sa
from alembic import op

revision = "a6d903fb124e"
down_revision = "9c7b8608e613"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "admission_score",
        "passing_score",
        existing_type=sa.Integer(),
        type_=sa.Float(),
        existing_nullable=True,
    )


def downgrade() -> None:
    # Refuse to silently round college scores when rolling back the schema.
    op.execute(
        """DO $$ BEGIN
        IF EXISTS (
            SELECT 1 FROM admission_score
            WHERE passing_score <> trunc(passing_score)
        ) THEN
            RAISE EXCEPTION 'Cannot downgrade fractional admission scores';
        END IF;
        END $$"""
    )
    op.alter_column(
        "admission_score",
        "passing_score",
        existing_type=sa.Float(),
        type_=sa.Integer(),
        existing_nullable=True,
        postgresql_using="passing_score::integer",
    )
