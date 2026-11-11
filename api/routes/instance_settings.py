import json
import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

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
from config.content_gate import content_write
from config.instance import SPECS, install
from config.model_settings import MODEL_FIELDS
from db.postgres.db import get_async_session
from db.postgres.services.instance_settings import (
    InstanceSettingsService,
    InstanceSnapshot,
    SettingsConflict,
)

router = APIRouter(
    prefix="/settings/instance",
    tags=["settings"],
    dependencies=[Depends(require_superadmin)],
)


async def describe(session: AsyncSession) -> dict:
    snapshot = await InstanceSettingsService(session).read()
    return describe_snapshot(snapshot)


def describe_snapshot(snapshot: InstanceSnapshot) -> dict:
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
@content_write()
async def update_instance_settings(
    data: SettingsChange, session: AsyncSession = Depends(get_async_session)
) -> dict:
    try:
        if MODEL_FIELDS.intersection(data.changes):
            raise ConfigurationError("Сохраните выбранные модели в разделе «Модели»")
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
        return describe_snapshot(updated)
    except SettingsConflict as exc:
        raise HTTPException(409, detail=str(exc)) from None
    except (ConfigurationError, ValueError) as exc:
        raise HTTPException(400, detail=str(exc)) from None


@router.post("/branding", response_model=InstanceSettingsResponse)
@content_write()
async def save_branding(
    revision: int = Form(...),
    changes: str = Form("{}", max_length=100_000),
    remove_logo: bool = Form(False),
    logo: UploadFile | None = File(None),
    session: AsyncSession = Depends(get_async_session),
) -> dict:
    from api.services.branding import MAX_BYTES, store_logo

    paths: list[Path] = []
    committed = False
    try:
        data = SettingsChange.model_validate(
            {"revision": revision, "changes": json.loads(changes)}
        )
        snapshot = await InstanceSettingsService(session).read()
        if snapshot.revision != revision:
            raise HTTPException(409, detail="Настройки изменены: обновите версию")
        if logo and remove_logo:
            raise HTTPException(400, detail="Выберите загрузку или удаление логотипа")
        if logo:
            content = await logo.read(MAX_BYTES + 1)
            url, paths = await run_in_threadpool(store_logo, content)
            data.changes["UNIVERSITY_LOGO_URL"] = url
        elif remove_logo:
            data.changes["UNIVERSITY_LOGO_URL"] = ""
        result = await update_instance_settings(data, session)
        committed = True
        return result
    except ValidationError:
        raise HTTPException(400, detail="Некорректные поля оформления") from None
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from None
    except OSError:
        raise HTTPException(503, detail="Хранилище логотипов недоступно") from None
    finally:
        if logo:
            await logo.close()
        if not committed:
            for path in paths:
                path.unlink(missing_ok=True)
