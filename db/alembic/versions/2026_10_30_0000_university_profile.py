"""Bootstrap a safe profile for empty installations, preserve existing instances."""

import json

import sqlalchemy as sa
from alembic import op

revision = "7b4e1d9c2a60"
down_revision = "5e7a9c1b3d2f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    key = "instance_settings_v1"
    raw = bind.execute(
        sa.text("SELECT value FROM settings WHERE key = :key"), {"key": key}
    ).scalar_one_or_none()
    if raw is not None:
        data = json.loads(raw)
        if "UNIVERSITY_PROFILE" in data["values"]:
            return
        data["values"]["UNIVERSITY_PROFILE"] = "nsu"
        data["revision"] += 1
        bind.execute(
            sa.text(
                "UPDATE settings SET value=:value, updated_at=CURRENT_TIMESTAMP "
                "WHERE key=:key"
            ),
            {"key": key, "value": json.dumps(data)},
        )
        return
    existing = bind.execute(
        sa.text("""
        SELECT EXISTS(SELECT 1 FROM faculty) OR EXISTS(SELECT 1 FROM "user")
          OR EXISTS(SELECT 1 FROM admins) OR EXISTS(SELECT 1 FROM faq_entry)
          OR EXISTS(SELECT 1 FROM document)
    """)
    ).scalar_one()
    data = {
        "revision": 1,
        "values": {"UNIVERSITY_PROFILE": "nsu" if existing else "custom"},
    }
    bind.execute(
        sa.text("""INSERT INTO settings (key, value, description, updated_at)
        VALUES (:key, :value, :description, CURRENT_TIMESTAMP)"""),
        {
            "key": key,
            "value": json.dumps(data),
            "description": "Начальный профиль вуза",
        },
    )


def downgrade() -> None:
    # Profile is harmless to the old application; never delete user settings.
    pass
