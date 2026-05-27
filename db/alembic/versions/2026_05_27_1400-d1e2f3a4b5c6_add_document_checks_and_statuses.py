"""Rename legacy document statuses

Revision ID: d1e2f3a4b5c6
Revises: c6d7e8f9a0b1
Create Date: 2026-05-27 14:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, Sequence[str], None] = "c6d7e8f9a0b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("UPDATE document SET status = 'indexed' WHERE status = 'active'")
    op.execute(
        "UPDATE document SET status = 'indexing_failed' WHERE status = 'error'"
    )


def downgrade() -> None:
    op.execute("UPDATE document SET status = 'active' WHERE status = 'indexed'")
    op.execute(
        "UPDATE document SET status = 'error' WHERE status = 'indexing_failed'"
    )
