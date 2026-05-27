"""Store document check state on document

Revision ID: e2f3a4b5c6d7
Revises: d1e2f3a4b5c6
Create Date: 2026-05-27 15:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e2f3a4b5c6d7"
down_revision: Union[str, Sequence[str], None] = "d1e2f3a4b5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("document", sa.Column("last_checked_at", sa.DateTime(), nullable=True))
    op.add_column(
        "document", sa.Column("last_checked_hash", sa.String(length=64), nullable=True)
    )
    op.add_column("document", sa.Column("last_check_message", sa.Text(), nullable=True))

    op.execute("UPDATE document SET status = 'проиндексирован' WHERE status = 'active'")
    op.execute("UPDATE document SET status = 'проиндексирован' WHERE status = 'indexed'")
    op.execute("UPDATE document SET status = 'изменён' WHERE status = 'changed'")
    op.execute(
        "UPDATE document SET status = 'источник недоступен' "
        "WHERE status = 'source_unavailable'"
    )
    op.execute("UPDATE document SET status = 'нет ссылки' WHERE status = 'no_source'")
    op.execute("UPDATE document SET status = 'удалён' WHERE status = 'deleted'")
    op.execute(
        "UPDATE document SET status = 'ошибка индексации' "
        "WHERE status IN ('error', 'indexing_failed')"
    )

    op.execute("DROP TABLE IF EXISTS document_check")


def downgrade() -> None:
    op.execute("UPDATE document SET status = 'indexed' WHERE status = 'проиндексирован'")
    op.execute("UPDATE document SET status = 'changed' WHERE status = 'изменён'")
    op.execute(
        "UPDATE document SET status = 'source_unavailable' "
        "WHERE status = 'источник недоступен'"
    )
    op.execute("UPDATE document SET status = 'no_source' WHERE status = 'нет ссылки'")
    op.execute(
        "UPDATE document SET status = 'indexing_failed' "
        "WHERE status = 'ошибка индексации'"
    )
    op.execute("UPDATE document SET status = 'deleted' WHERE status = 'удалён'")

    op.drop_column("document", "last_check_message")
    op.drop_column("document", "last_checked_hash")
    op.drop_column("document", "last_checked_at")
