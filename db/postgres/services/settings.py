from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.models import Settings


@dataclass(frozen=True)
class RateLimitSettingsValues:
    system_requests_per_day: int = 10000
    user_requests_per_day: int = 100


class SettingsService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_rate_limit_settings(self) -> RateLimitSettingsValues:
        settings = await self._get_many(
            [
                "rate_limit_system_requests_per_day",
                "rate_limit_user_requests_per_day",
            ]
        )
        defaults = RateLimitSettingsValues()
        return RateLimitSettingsValues(
            system_requests_per_day=self._int_value(
                settings.get("rate_limit_system_requests_per_day"),
                defaults.system_requests_per_day,
            ),
            user_requests_per_day=self._int_value(
                settings.get("rate_limit_user_requests_per_day"),
                defaults.user_requests_per_day,
            ),
        )

    async def update_rate_limit_settings(
        self,
        system_requests_per_day: int,
        user_requests_per_day: int,
    ) -> RateLimitSettingsValues:
        await self._upsert(
            "rate_limit_system_requests_per_day",
            str(system_requests_per_day),
            "Количество запросов в день для всей системы",
        )
        await self._upsert(
            "rate_limit_user_requests_per_day",
            str(user_requests_per_day),
            "Количество запросов в день для одного пользователя",
        )
        await self.session.commit()
        return await self.get_rate_limit_settings()

    async def _get_many(self, keys: list[str]) -> dict[str, Settings]:
        result = await self.session.execute(
            select(Settings).where(Settings.key.in_(keys))
        )
        return {setting.key: setting for setting in result.scalars().all()}

    async def _upsert(self, key: str, value: str, description: str) -> None:
        result = await self.session.execute(
            select(Settings).where(Settings.key == key)
        )
        setting = result.scalar_one_or_none()
        if setting is None:
            self.session.add(
                Settings(key=key, value=value, description=description)
            )
            return

        setting.value = value
        setting.description = description

    @staticmethod
    def _int_value(setting: Settings | None, default: int) -> int:
        if setting is None:
            return default
        try:
            return int(setting.value)
        except ValueError:
            return default
