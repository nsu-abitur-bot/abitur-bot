"""rename_snils_to_applicant_id_add_status

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-02 00:01:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Переименовать snils_id → applicant_id и уменьшить размер до 7
    op.alter_column(
        "user",
        "snils_id",
        new_column_name="applicant_id",
        existing_type=sa.String(length=14),
        type_=sa.String(length=7),
        existing_nullable=True,
        nullable=True,
    )

    # Добавить поле status в user_rating
    op.add_column(
        "user_rating",
        sa.Column(
            "status",
            sa.String(length=100),
            nullable=False,
            server_default="",
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("user_rating", "status")

    op.alter_column(
        "user",
        "applicant_id",
        new_column_name="snils_id",
        existing_type=sa.String(length=7),
        type_=sa.String(length=14),
        existing_nullable=True,
        nullable=True,
    )
