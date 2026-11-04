"""Merge answer feedback and FAQ verification migration heads."""

revision = "c9d2e4f6a801"
down_revision = ("8f31c9d1a204", "a4c8e1f3b7d2")
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Join both histories without modifying their data or schemas."""
    pass


def downgrade() -> None:
    """Restore both parent heads without reverting either feature."""
    pass
