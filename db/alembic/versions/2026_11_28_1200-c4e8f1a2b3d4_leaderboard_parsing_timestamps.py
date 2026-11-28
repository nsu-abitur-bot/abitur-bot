"""Add last_parsed_at and data_updated_at to leaderboard."""

import sqlalchemy as sa
from alembic import op

revision = "c4e8f1a2b3d4"
down_revision = "fac64d37b12e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "leaderboard", sa.Column("last_parsed_at", sa.DateTime(), nullable=True)
    )
    op.add_column(
        "leaderboard", sa.Column("data_updated_at", sa.DateTime(), nullable=True)
    )
    op.create_index("idx_leaderboards_last_parsed", "leaderboard", ["last_parsed_at"])


def downgrade() -> None:
    op.drop_index("idx_leaderboards_last_parsed", table_name="leaderboard")
    op.drop_column("leaderboard", "data_updated_at")
    op.drop_column("leaderboard", "last_parsed_at")
