"""Two model assignments and explicit durable embedding migration controls."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException

from api.auth.dependencies import require_superadmin
from api.routes.instance_settings import describe_snapshot
from api.schemas.instance_settings import SettingsChange
from api.services.instance_settings import merge_changes
from api.services.model_migration import atomic_json, start_migration
from config.content_gate import ModelMigrationBusy, index_fence, migration_marker
from config.model_settings import MODEL_VERSION
from db.postgres.db import AsyncSessionLocal
from db.postgres.services.instance_settings import (
    InstanceSettingsService,
    SettingsConflict,
)
from db.postgres.services.model_migration import ACTIVE_STATES, ModelMigrationService

router = APIRouter(
    prefix="/settings/instance/models",
    tags=["settings"],
    dependencies=[Depends(require_superadmin)],
)


class ModelSettingsChange(SettingsChange):
    confirm_reindex: bool = False


@router.get("")
async def model_configuration() -> dict:
    async with AsyncSessionLocal() as session:
        snapshot = await InstanceSettingsService(session).read()
        values = merge_changes(snapshot, {})
        text, embedding = values["LLM_PROVIDER"], values["FAQ_EMBEDDING_PROVIDER"]
        legacy = [
            {
                "label": label,
                "provider": values[key],
                "model": values.get(values[key].upper() + suffix, ""),
            }
            for label, key, suffix in (
                ("Ответы", "LLM_PROVIDER", "_MODEL"),
                ("Граф", "LIGHTRAG_LLM_PROVIDER", "_GRAPH_MODEL"),
                ("FAQ", "FAQ_EMBEDDING_PROVIDER", "_EMBEDDING_MODEL"),
                ("База знаний", "LIGHTRAG_EMBEDDING_PROVIDER", "_GRAPH_EMBEDDING_MODEL"),
                ("Распознавание документов", "PDF_PARSER_PROVIDER", "_MODEL_VISION"),
            )
        ]
        for row in legacy:
            if row["provider"] == "gigachat":
                row["model"] = values["GIGACHAT_EMBEDDING_MODEL"]
        job = await ModelMigrationService(session).active()
        return {
            "snapshot": describe_snapshot(snapshot),
            "revision": snapshot.revision,
            "version": int(values.get(MODEL_VERSION, "1")),
            "assignments": {
                "TEXT_PROVIDER": text,
                "TEXT_MODEL": values[text.upper() + "_MODEL"],
                "EMBEDDING_PROVIDER": embedding,
                "EMBEDDING_MODEL": values[embedding.upper() + "_EMBEDDING_MODEL"],
            },
            "legacy": legacy if values.get(MODEL_VERSION) != "2" else [],
            "job": ModelMigrationService.public(job) if job else None,
        }


@router.post("")
async def save_models(data: ModelSettingsChange) -> dict:
    try:
        return await start_migration(
            data.revision, data.changes, confirm_reindex=data.confirm_reindex
        )
    except SettingsConflict as exc:
        raise HTTPException(409, detail=str(exc)) from None
    except ModelMigrationBusy as exc:
        raise HTTPException(423, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from None
    except Exception:
        raise HTTPException(
            400,
            detail="Не удалось проверить подключение моделей; "
            "рабочие настройки сохранены",
        ) from None


@router.post("/job/{job_id}/{action}")
async def control_job(job_id: str, action: Literal["retry", "cancel"]) -> dict:
    try:
        async with index_fence(exclusive=True):
            async with AsyncSessionLocal() as session:
                service = ModelMigrationService(session)
                await service.lock()
                active = await service.active()
                if not active or active["id"] != job_id:
                    raise HTTPException(
                        404, detail="Задание не найдено или заменено новым"
                    )
                job = active
                if action == "cancel":
                    if job["status"] in ACTIVE_STATES:
                        job["status"] = "cancelling"
                else:
                    snapshot = await InstanceSettingsService(session).read()
                    if snapshot.revision != job["revision"]:
                        raise SettingsConflict(
                            "Настройки изменились; создайте новое задание"
                        )
                    if job["status"] not in {"failed", "cancelled"}:
                        return service.public(job)
                    job["status"], job["error"] = "queued", None

                await service.save(job)
                await session.commit()
                if action == "retry" and job["status"] == "queued":
                    atomic_json(migration_marker(), {"id": job["id"]})
                return service.public(job)
    except SettingsConflict as exc:
        raise HTTPException(409, detail=str(exc)) from None
