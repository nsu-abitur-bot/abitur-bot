"""add_index_on_applicant_id

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-03-02 00:03:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: create index on applicant_id."""
    op.create_index(
        "idx_user_applicant_id", "user", ["applicant_id"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema: drop index on applicant_id."""
    op.drop_index("idx_user_applicant_id", table_name="user")
