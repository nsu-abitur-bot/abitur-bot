import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import require_superadmin
from api.schemas.instance_settings import (
    ConnectionCheck,
    ConnectionCheckResponse,
    InstanceSettingsResponse,
    SettingsChange,
)
from api.services.instance_settings import (
    ConfigurationError,
    check_connection,
    guard_embeddings,
    merge_changes,
    required_checks,
)
from config.instance import SPECS, install
from db.postgres.db import get_async_session
from db.postgres.services.instance_settings import (
    InstanceSettingsService,
    SettingsConflict,
)

router = APIRouter(
    prefix="/settings/instance",
    tags=["settings"],
    dependencies=[Depends(require_superadmin)],
)


async def describe(session: AsyncSession) -> dict:
    snapshot = await InstanceSettingsService(session).read()
    return {
        "revision": snapshot.revision,
        "encryption_ready": bool(os.getenv("SETTINGS_ENCRYPTION_KEY")),
        "settings": [
            {
                "key": key,
                "label": spec.label,
                "group": spec.group,
                "kind": spec.kind,
                "choices": spec.choices,
                "secret": spec.secret,
                "configured": bool(snapshot.values.get(key, spec.default)),
                "value": None if spec.secret else snapshot.values.get(key, spec.default),
            }
            for key, spec in SPECS.items()
        ],
    }


@router.get("", response_model=InstanceSettingsResponse)
async def get_instance_settings(
    session: AsyncSession = Depends(get_async_session),
) -> dict:
    try:
        return await describe(session)
    except ValueError as exc:
        raise HTTPException(503, detail=str(exc)) from None


@router.post("/check", response_model=ConnectionCheckResponse)
async def test_connection(
    data: ConnectionCheck, session: AsyncSession = Depends(get_async_session)
) -> dict:
    try:
        snapshot = await InstanceSettingsService(session).read()
        if data.revision != snapshot.revision:
            raise SettingsConflict("Настройки изменены: обновите страницу")
        values = merge_changes(snapshot, data.changes)
        return {"message": await check_connection(data.target, values)}
    except SettingsConflict as exc:
        raise HTTPException(409, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from None


@router.put("", response_model=InstanceSettingsResponse)
async def update_instance_settings(
    data: SettingsChange, session: AsyncSession = Depends(get_async_session)
) -> dict:
    try:
        service = InstanceSettingsService(session)
        snapshot = await service.read()
        if snapshot.revision != data.revision:
            raise SettingsConflict("Настройки изменены: обновите страницу")
        values = merge_changes(snapshot, data.changes)
        await guard_embeddings(session, snapshot, values)
        # No database locks while awaiting remote services.
        for target in sorted(required_checks(data.changes, values)):
            await check_connection(target, values)
        await guard_embeddings(session, snapshot, values, lock=True)
        updated = await service.write(values, data.revision)
        install(updated.values, updated.revision)
        return await describe(session)
    except SettingsConflict as exc:
        raise HTTPException(409, detail=str(exc)) from None
    except (ConfigurationError, ValueError) as exc:
        raise HTTPException(400, detail=str(exc)) from None
