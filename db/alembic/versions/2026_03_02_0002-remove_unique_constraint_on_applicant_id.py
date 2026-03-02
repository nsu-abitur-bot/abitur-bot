"""remove_unique_constraint_on_applicant_id

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-03-02 00:02:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: remove unique constraint on applicant_id."""
    op.drop_constraint("uq_user_applicant_id", "user", type_="unique", if_exists=True)


def downgrade() -> None:
    """Downgrade schema: re-create unique constraint on applicant_id."""
    op.create_unique_constraint("uq_user_applicant_id", "user", ["applicant_id"])
