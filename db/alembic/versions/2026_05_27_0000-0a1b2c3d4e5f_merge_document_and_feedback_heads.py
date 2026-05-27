"""merge document and feedback migration heads

Revision ID: 0a1b2c3d4e5f
Revises: 4b5c6d7e8f90, 9f1a2b3c4d5e
Create Date: 2026-05-27 00:00:00.000000

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "0a1b2c3d4e5f"
down_revision: Union[str, Sequence[str], None] = (
    "4b5c6d7e8f90",
    "9f1a2b3c4d5e",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
