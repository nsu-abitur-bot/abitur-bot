"""Store announced seat counts for admission records.

Revision ID: 0af365dca483
Revises: 0af365dca482
"""

import sqlalchemy as sa
from alembic import op

revision = "0af365dca483"
down_revision = "0af365dca482"
branch_labels = None
depends_on = None

_TABLE = "admission_score"
_CONSTRAINT = "ck_admission_score_seat_count"


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column("seat_count", sa.Integer(), nullable=True))
    op.create_check_constraint(_CONSTRAINT, _TABLE, "seat_count >= 0")


def downgrade() -> None:
    op.execute("LOCK TABLE admission_score IN ACCESS EXCLUSIVE MODE")
    op.execute(
        """DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM admission_score WHERE seat_count IS NOT NULL) THEN
            RAISE EXCEPTION 'Cannot downgrade admission records with seat counts';
        END IF;
        END $$"""
    )
    op.drop_constraint(_CONSTRAINT, _TABLE, type_="check")
    op.drop_column(_TABLE, "seat_count")
