"""Add telegram_id and max_id to user

Revision ID: 8d2b3f4a1c90
Revises: 8c7d6e5a4b3c
Create Date: 2026-04-09 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8d2b3f4a1c90"
down_revision: Union[str, Sequence[str], None] = "8c7d6e5a4b3c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Ensure internal user_id can be generated independently from messenger IDs.
    op.execute('CREATE SEQUENCE IF NOT EXISTS user_user_id_seq OWNED BY "user".user_id')
    op.execute(
        "SELECT setval('user_user_id_seq', "
        'COALESCE((SELECT MAX(user_id) FROM "user"), 0) + 1, false)'
    )
    op.alter_column(
        "user",
        "user_id",
        existing_type=sa.BigInteger(),
        server_default=sa.text("nextval('user_user_id_seq')"),
        existing_nullable=False,
    )

    op.add_column("user", sa.Column("telegram_id", sa.BigInteger(), nullable=True))
    op.add_column("user", sa.Column("max_id", sa.String(length=255), nullable=True))

    # Backfill existing users: old user_id values were Telegram IDs.
    op.execute('UPDATE "user" SET telegram_id = user_id WHERE telegram_id IS NULL')

    op.create_index("ux_user_telegram_id", "user", ["telegram_id"], unique=True)
    op.create_index("ux_user_max_id", "user", ["max_id"], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ux_user_max_id", table_name="user")
    op.drop_index("ux_user_telegram_id", table_name="user")
    op.drop_column("user", "max_id")
    op.drop_column("user", "telegram_id")
    op.alter_column(
        "user",
        "user_id",
        existing_type=sa.BigInteger(),
        server_default=None,
        existing_nullable=False,
    )
    op.execute("DROP SEQUENCE IF EXISTS user_user_id_seq")
