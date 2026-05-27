from pydantic import BaseModel, Field


class RateLimitSettings(BaseModel):
    """Настройки дневных лимитов запросов."""

    system_requests_per_day: int = Field(
        10000,
        ge=1,
        le=10000000,
        description="Количество запросов в день для всей системы",
    )
    user_requests_per_day: int = Field(
        100,
        ge=1,
        le=100000,
        description="Количество запросов в день для одного пользователя",
    )


class RateLimitSettingsUpdate(BaseModel):
    """Обновление дневных лимитов запросов."""

    system_requests_per_day: int = Field(ge=1, le=10000000)
    user_requests_per_day: int = Field(ge=1, le=100000)
