"""add_index_on_created_at

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-03-02 00:04:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: create index on user.created_at for stats queries."""
    op.create_index("idx_user_created_at", "user", ["created_at"], unique=False)


def downgrade() -> None:
    """Downgrade schema: drop index on user.created_at."""
    op.drop_index("idx_user_created_at", table_name="user")
