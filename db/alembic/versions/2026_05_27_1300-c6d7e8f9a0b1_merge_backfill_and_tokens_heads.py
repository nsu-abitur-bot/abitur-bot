"""merge backfill and tokens heads

Revision ID: c6d7e8f9a0b1
Revises: b1c2d3e4f5a6, 7f9a2b1c4d5e
Create Date: 2026-05-27 13:00:00.000000

"""

from typing import Sequence, Union

revision: str = "c6d7e8f9a0b1"
down_revision: Union[str, Sequence[str], None] = (
    "b1c2d3e4f5a6",
    "7f9a2b1c4d5e",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
