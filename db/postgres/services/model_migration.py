"""Durable, encrypted migration jobs in the existing settings journal."""

import json
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Settings
from .instance_settings import cipher

ACTIVE_KEY = "model_migration_active"
ACTIVE_STATES = {"queued", "running", "cancelling"}


class ModelMigrationService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def lock(self) -> None:
        await self.session.execute(text("SELECT pg_advisory_xact_lock(46, 1)"))

    async def get(self, job_id: str) -> dict[str, Any] | None:
        value = await self.session.scalar(
            select(Settings.value).where(Settings.key == "model_migration:" + job_id)
        )
        return json.loads(value) if value else None

    async def active(self) -> dict[str, Any] | None:
        job_id = await self.session.scalar(
            select(Settings.value).where(Settings.key == ACTIVE_KEY)
        )
        return await self.get(job_id) if job_id else None

    async def save(self, job: dict[str, Any]) -> None:
        key = "model_migration:" + job["id"]
        row = await self.session.scalar(select(Settings).where(Settings.key == key))
        if row is None:
            self.session.add(
                Settings(
                    key=key,
                    value=json.dumps(job),
                    description="Пересчёт эмбеддингов; целевые настройки зашифрованы",
                )
            )
        else:
            row.value = json.dumps(job)

    async def create(self, job: dict[str, Any], values: dict[str, str]) -> None:
        job["encrypted_target"] = cipher().encrypt(json.dumps(values).encode()).decode()
        await self.save(job)
        row = await self.session.scalar(
            select(Settings).where(Settings.key == ACTIVE_KEY)
        )
        if row is None:
            self.session.add(
                Settings(
                    key=ACTIVE_KEY,
                    value=job["id"],
                    description="Последнее задание пересчёта",
                )
            )
        else:
            row.value = job["id"]

    @staticmethod
    def target(job: dict[str, Any]) -> dict[str, str]:
        return json.loads(cipher().decrypt(job["encrypted_target"].encode()))

    @staticmethod
    def public(job: dict[str, Any]) -> dict[str, Any]:
        result = {
            key: job.get(key)
            for key in (
                "id",
                "status",
                "revision",
                "progress",
                "error",
                "completed_revision",
            )
        }
        target = ModelMigrationService.target(job)
        result["target_models"] = {
            key: target[key]
            for key in (
                "TEXT_PROVIDER",
                "TEXT_MODEL",
                "EMBEDDING_PROVIDER",
                "EMBEDDING_MODEL",
            )
        }
        return result
