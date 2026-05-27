"""add tokens_used to message_logs

Revision ID: 7f9a2b1c4d5e
Revises: d267a76e1a85
Create Date: 2026-05-27 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7f9a2b1c4d5e"
down_revision: Union[str, Sequence[str], None] = "d267a76e1a85"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "message_logs",
        sa.Column("tokens_used", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ix_message_logs_tokens_used",
        "message_logs",
        ["tokens_used"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_message_logs_tokens_used", table_name="message_logs")
    op.drop_column("message_logs", "tokens_used")
