from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.crag import CragSettings, CragSettingsUpdate
from api.schemas.rate_limit import RateLimitSettings, RateLimitSettingsUpdate
from db.postgres.db import get_async_session
from db.postgres.services.settings import SettingsService
from rag.crag import load_crag_config

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


def _crag_config_to_schema(cfg) -> CragSettings:
    return CragSettings(
        enabled=cfg.enabled,
        relevance_threshold=cfg.relevance_threshold,
        min_chunks=cfg.min_chunks,
        allow_refine=cfg.allow_refine,
        use_faculty_table=cfg.use_faculty_table,
        max_graded_chunks=cfg.max_graded_chunks,
    )


@router.get(
    "/crag",
    response_model=CragSettings,
    summary="Получить настройки корректирующего RAG (CRAG)",
)
async def get_crag_settings() -> CragSettings:
    try:
        cfg = await load_crag_config()
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return _crag_config_to_schema(cfg)


@router.put(
    "/crag",
    response_model=CragSettings,
    summary="Обновить настройки корректирующего RAG (CRAG)",
)
async def update_crag_settings(
    data: CragSettingsUpdate,
    service: SettingsService = Depends(get_settings_service),
) -> CragSettings:
    try:
        await service.update_crag_settings(
            enabled=data.enabled,
            relevance_threshold=data.relevance_threshold,
            min_chunks=data.min_chunks,
            allow_refine=data.allow_refine,
            use_faculty_table=data.use_faculty_table,
            max_graded_chunks=data.max_graded_chunks,
        )
        cfg = await load_crag_config()
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return _crag_config_to_schema(cfg)
