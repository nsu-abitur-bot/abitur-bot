"""Encrypted application configuration, committed as one versioned snapshot."""

import json
import os
from dataclasses import dataclass

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from config.instance import SPECS

from ..models import Settings

SETTINGS_KEY = "instance_settings_v1"


@dataclass(frozen=True)
class InstanceSnapshot:
    revision: int
    values: dict[str, str]


class SettingsConflict(ValueError):
    pass


def cipher() -> Fernet:
    key = os.getenv("SETTINGS_ENCRYPTION_KEY", "")
    try:
        return Fernet(key.encode())
    except (ValueError, TypeError):
        raise ValueError("Задайте SETTINGS_ENCRYPTION_KEY (Fernet) в окружении") from None


class InstanceSettingsService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def read(self) -> InstanceSnapshot:
        result = await self.session.execute(
            select(Settings.value).where(Settings.key == SETTINGS_KEY)
        )
        raw = result.scalar_one_or_none()
        if raw is None:
            return InstanceSnapshot(0, {})
        data = json.loads(raw)
        values = data["values"]
        try:
            for name, spec in SPECS.items():
                if spec.secret and values.get(name):
                    values[name] = cipher().decrypt(values[name].encode()).decode()
        except InvalidToken:
            raise ValueError(
                "Не удалось расшифровать настройки: проверьте мастер-ключ"
            ) from None
        return InstanceSnapshot(data["revision"], values)

    async def write(
        self, values: dict[str, str], expected_revision: int
    ) -> InstanceSnapshot:
        # Serializes the initial insert as well as updates, across API workers.
        await self.session.execute(text("SELECT pg_advisory_xact_lock(356, 1)"))
        current = await self.read()
        if current.revision != expected_revision:
            raise SettingsConflict("Настройки уже изменены: обновите страницу")
        encrypted = dict(values)
        for name, spec in SPECS.items():
            if spec.secret and encrypted.get(name):
                encrypted[name] = cipher().encrypt(encrypted[name].encode()).decode()
        row = await self.session.scalar(
            select(Settings).where(Settings.key == SETTINGS_KEY)
        )
        raw = json.dumps({"revision": current.revision + 1, "values": encrypted})
        if row is None:
            self.session.add(
                Settings(
                    key=SETTINGS_KEY,
                    value=raw,
                    description="Настройки экземпляра; секреты зашифрованы",
                )
            )
        else:
            row.value = raw
        await self.session.commit()
        return InstanceSnapshot(current.revision + 1, values)
