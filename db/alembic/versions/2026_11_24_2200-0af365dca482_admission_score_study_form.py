"""Separate study form from admission basis in historical scores.

Revision ID: 0af365dca482
Revises: b8f2a9c4d6e1
"""

import sqlalchemy as sa
from alembic import op

revision = "0af365dca482"
down_revision = "b8f2a9c4d6e1"
branch_labels = None
depends_on = None

_TABLE = "admission_score"
_OLD_CONSTRAINT = "uq_admission_score_program_year_form"
_NEW_CONSTRAINT = "uq_admission_score_program_year_study_form_basis"


def upgrade() -> None:
    op.drop_constraint(_OLD_CONSTRAINT, _TABLE, type_="unique")
    op.alter_column(
        _TABLE, "form", new_column_name="admission_basis", existing_type=sa.String(20)
    )
    op.add_column(_TABLE, sa.Column("study_form", sa.String(20), nullable=True))
    op.create_unique_constraint(
        _NEW_CONSTRAINT,
        _TABLE,
        ["program_id", "year", "study_form", "admission_basis"],
        postgresql_nulls_not_distinct=True,
    )


def downgrade() -> None:
    op.execute("LOCK TABLE admission_score IN ACCESS EXCLUSIVE MODE")
    op.execute(
        """DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM admission_score WHERE study_form IS NOT NULL) THEN
            RAISE EXCEPTION 'Cannot downgrade classified admission scores';
        END IF;
        END $$"""
    )
    op.drop_constraint(_NEW_CONSTRAINT, _TABLE, type_="unique")
    op.drop_column(_TABLE, "study_form")
    op.alter_column(
        _TABLE, "admission_basis", new_column_name="form", existing_type=sa.String(20)
    )
    op.create_unique_constraint(_OLD_CONSTRAINT, _TABLE, ["program_id", "year", "form"])
