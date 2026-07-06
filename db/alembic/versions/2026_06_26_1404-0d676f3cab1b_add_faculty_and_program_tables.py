"""add faculty and program tables

Revision ID: 0d676f3cab1b
Revises: a1b2c3d4e6f0
Create Date: 2026-06-26 14:04:29.484594

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0d676f3cab1b"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e6f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "faculty",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=300), nullable=False),
        sa.Column("aliases", sa.JSON(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_faculty_is_active", "faculty", ["is_active"], unique=False)
    op.create_index("ix_faculty_name", "faculty", ["name"], unique=False)
    op.create_table(
        "program",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("faculty_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=300), nullable=False),
        sa.Column("code", sa.String(length=20), nullable=True),
        sa.Column("level", sa.String(length=20), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["faculty_id"], ["faculty.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "faculty_id", "name", "level", name="uq_program_faculty_name_level"
        ),
    )
    op.create_index("ix_program_faculty_id", "program", ["faculty_id"], unique=False)
    op.create_index("ix_program_level", "program", ["level"], unique=False)
    op.create_index("ix_program_name", "program", ["name"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_program_name", table_name="program")
    op.drop_index("ix_program_level", table_name="program")
    op.drop_index("ix_program_faculty_id", table_name="program")
    op.drop_table("program")
    op.drop_index("ix_faculty_name", table_name="faculty")
    op.drop_index("ix_faculty_is_active", table_name="faculty")
    op.drop_table("faculty")
