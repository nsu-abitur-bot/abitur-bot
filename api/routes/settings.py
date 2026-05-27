from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.rate_limit import RateLimitSettings, RateLimitSettingsUpdate
from db.postgres.db import get_async_session
from db.postgres.services.settings import SettingsService

router = APIRouter(prefix="/settings", tags=["settings"])


def get_settings_service(
    session: AsyncSession = Depends(get_async_session),
) -> SettingsService:
    return SettingsService(session)


@router.get(
    "/rate-limit",
    response_model=RateLimitSettings,
    summary="Получить настройки дневных лимитов запросов",
)
async def get_rate_limit_settings(
    service: SettingsService = Depends(get_settings_service),
) -> RateLimitSettings:
    try:
        settings = await service.get_rate_limit_settings()
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return RateLimitSettings(
        system_requests_per_day=settings.system_requests_per_day,
        user_requests_per_day=settings.user_requests_per_day,
    )


@router.put(
    "/rate-limit",
    response_model=RateLimitSettings,
    summary="Обновить настройки дневных лимитов запросов",
)
async def update_rate_limit_settings(
    data: RateLimitSettingsUpdate,
    service: SettingsService = Depends(get_settings_service),
) -> RateLimitSettings:
    try:
        settings = await service.update_rate_limit_settings(
            system_requests_per_day=data.system_requests_per_day,
            user_requests_per_day=data.user_requests_per_day,
        )
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return RateLimitSettings(
        system_requests_per_day=settings.system_requests_per_day,
        user_requests_per_day=settings.user_requests_per_day,
    )
