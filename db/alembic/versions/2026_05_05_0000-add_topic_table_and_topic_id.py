"""Add topic table and topic_id to message_logs

Revision ID: 3a4b5c6d7e8f
Revises: 2a3b4c5d6e7f
Create Date: 2026-05-05 00:00:00.000000

"""

from datetime import datetime
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "3a4b5c6d7e8f"
down_revision: Union[str, None] = "2a3b4c5d6e7f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "topic",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("label"),
    )

    op.add_column("message_logs", sa.Column("topic_id", sa.BigInteger(), nullable=True))
    op.create_foreign_key(
        "fk_message_logs_topic_id",
        "message_logs",
        "topic",
        ["topic_id"],
        ["id"],
    )
    op.create_index("ix_message_logs_topic_id", "message_logs", ["topic_id"])

    # Вставляем начальные темы
    op.bulk_insert(
        sa.table(
            "topic",
            sa.Column("id", sa.BigInteger()),
            sa.Column("label", sa.String(255)),
            sa.Column("description", sa.Text()),
            sa.Column("is_active", sa.Boolean()),
            sa.Column("created_at", sa.DateTime()),
        ),
        [
            {"id": 1, "label": "Общежитие", "description": "Вопросы об общежитии",
             "is_active": True, "created_at": datetime.utcnow()},
            {"id": 2,
             "label": "Проходные баллы",
             "description": "Вопросы о проходных баллах",
             "is_active": True, "created_at": datetime.utcnow()},
            {"id": 3, "label": "Документы", "description": "Вопросы о документах",
             "is_active": True, "created_at": datetime.utcnow()},
            {"id": 4,
             "label": "Специальности",
             "description": "Вопросы о специальностях",
             "is_active": True, "created_at": datetime.utcnow()},
            {"id": 5,
             "label": "Поступление",
             "description": "Общие вопросы о поступлении",
             "is_active": True, "created_at": datetime.utcnow()},
        ]
    )


def downgrade() -> None:
    op.drop_index("ix_message_logs_topic_id", table_name="message_logs")
    op.drop_constraint("fk_message_logs_topic_id", "message_logs", type_="foreignkey")
    op.drop_column("message_logs", "topic_id")
    op.drop_table("topic")