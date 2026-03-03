"""remove_unique_constraint_on_applicant_id

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-03-02 00:02:00.000000

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: remove unique constraint on applicant_id if exists."""
    # Используем PostgreSQL-специфичный синтаксис IF EXISTS
    op.execute(
        text('ALTER TABLE "user" DROP CONSTRAINT IF EXISTS uq_user_applicant_id')
    )


def downgrade() -> None:
    """Downgrade schema: re-create unique constraint on applicant_id."""
    op.create_unique_constraint("uq_user_applicant_id", "user", ["applicant_id"])
