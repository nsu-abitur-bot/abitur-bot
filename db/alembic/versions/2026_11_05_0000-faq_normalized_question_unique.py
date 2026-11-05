"""Add normalized question key with unique index to FAQ entries (#295)."""

import sqlalchemy as sa
from alembic import op

revision = "b8f2a9c4d6e1"
down_revision = "9c7b8608e613"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("faq_entry", sa.Column("normalized_question", sa.Text(), nullable=True))
    # Бэкфилл: тот же результат, что normalize_question() в ORM.
    op.execute(
        r"UPDATE faq_entry SET normalized_question = "
        r"lower(regexp_replace("
        r"regexp_replace(question, '^\s+|\s+$', '', 'g'), '\s+', ' ', 'g'))"
    )
    # Дубли, накопленные багом (#295): оставляем самую свежую запись.
    op.execute(
        r"DELETE FROM faq_entry a USING faq_entry b "
        r"WHERE a.normalized_question = b.normalized_question "
        r"AND (a.updated_at, a.created_at, a.id) < (b.updated_at, b.created_at, b.id)"
    )
    op.alter_column(
        "faq_entry", "normalized_question", existing_type=sa.Text(), nullable=False
    )
    op.create_index(
        "uq_faq_entry_normalized_question",
        "faq_entry",
        ["normalized_question"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_faq_entry_normalized_question", table_name="faq_entry")
    op.drop_column("faq_entry", "normalized_question")
