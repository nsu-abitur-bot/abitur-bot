"""add admission_score table

Revision ID: c2f337939699
Revises: 0d676f3cab1b
Create Date: 2026-07-11 10:41:18.718314

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c2f337939699"
down_revision: Union[str, Sequence[str], None] = "0d676f3cab1b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "admission_score",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("program_id", sa.String(length=36), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("form", sa.String(length=20), nullable=False),
        sa.Column("passing_score", sa.Integer(), nullable=True),
        sa.Column("average_score", sa.Float(), nullable=True),
        sa.Column("source_url", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["program_id"], ["program.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "program_id",
            "year",
            "form",
            name="uq_admission_score_program_year_form",
        ),
    )
    op.create_index(
        "ix_admission_score_program", "admission_score", ["program_id"], unique=False
    )
    op.create_index("ix_admission_score_year", "admission_score", ["year"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_admission_score_year", table_name="admission_score")
    op.drop_index("ix_admission_score_program", table_name="admission_score")
    op.drop_table("admission_score")
