"""Merge document URL uniqueness and FAQ verification migration heads."""

revision = "9c7b8608e613"
down_revision = ("c3a4e7b92f10", "c9d2e4f6a801")
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Join both histories without modifying their data or schemas."""
    pass


def downgrade() -> None:
    """Restore both parent heads without reverting either feature."""
    pass
