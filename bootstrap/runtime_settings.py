import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from config.instance import SPECS, install
from db.postgres.db import AsyncSessionLocal, engine
from db.postgres.models import Settings
from db.postgres.services.instance_settings import InstanceSettingsService
from db.postgres.services.settings import SettingsService

logger = logging.getLogger(__name__)


async def refresh_settings() -> None:
    async with AsyncSessionLocal() as session:
        snapshot = await InstanceSettingsService(session).read()
    install(snapshot.values, snapshot.revision)


async def watch_settings() -> None:
    while True:
        await asyncio.sleep(2)
        try:
            await refresh_settings()
        except Exception:
            # Keep the last usable snapshot; never log decrypted configuration.
            logger.error("Не удалось обновить настройки экземпляра")


@asynccontextmanager
async def runtime_settings() -> AsyncIterator[None]:
    # A missing database/key must not enable legacy env tokens while the API
    # is recovering. Start with disabled channels and retry in the watcher.
    install({}, 0, loaded=False)
    try:
        await refresh_settings()
    except Exception:
        logger.error("Настройки недоступны при старте; подключения отключены")
    task = asyncio.create_task(watch_settings())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await engine.dispose()


async def import_environment() -> None:
    """Explicit one-time migration, never performed implicitly at API startup."""
    async with AsyncSessionLocal() as session:
        service = InstanceSettingsService(session)
        snapshot = await service.read()
        if snapshot.revision:
            raise ValueError("Настройки уже сохранены; импорт не перезаписывает их")
        values = {name: os.getenv(name, spec.default) for name, spec in SPECS.items()}
        values["FAQ_EMBEDDING_PROVIDER"] = os.getenv(
            "FAQ_EMBEDDING_PROVIDER", values["LLM_PROVIDER"]
        )
        for name, spec in SPECS.items():
            if spec.kind == "boolean":
                values[name] = (
                    "true"
                    if values[name].lower() in {"true", "1", "yes", "on"}
                    else "false"
                )
        values["TELEGRAM_ENABLED"] = "true" if values["BOT_TOKEN"] else "false"
        values["MAX_ENABLED"] = "true" if values["MAX_BOT_TOKEN"] else "false"
        values["LIGHTRAG_EMBEDDING_PROVIDER"] = os.getenv(
            "LIGHTRAG_EMBEDDING_PROVIDER", values["LIGHTRAG_LLM_PROVIDER"]
        )
        for old, new in (
            ("max_document_size", "MAX_DOCUMENT_SIZE"),
            ("max_document_pages", "MAX_DOCUMENT_PAGES"),
        ):
            stored = await SettingsService(session).get_value(old)
            if stored is not None:
                values[new] = stored
        for old, new in (
            ("FAQ_SIMILARITY_THRESHOLD", "faq_similarity_threshold"),
            ("CRAG_ENABLED", "crag_enabled"),
            ("CRAG_RELEVANCE_THRESHOLD", "crag_relevance_threshold"),
            ("CRAG_MIN_CHUNKS", "crag_min_chunks"),
            ("CRAG_ALLOW_REFINE", "crag_allow_refine"),
            ("CRAG_USE_FACULTY_TABLE", "crag_use_faculty_table"),
            ("CRAG_MAX_GRADED_CHUNKS", "crag_max_graded_chunks"),
        ):
            if (
                os.getenv(old) is not None
                and await SettingsService(session).get_value(new) is None
            ):
                session.add(
                    Settings(
                        key=new, value=os.environ[old], description="Перенесено из env"
                    )
                )
        await service.write(values, 0)
    print(
        "Настройки импортированы; секреты зашифрованы. "
        "Можно удалить прикладные env-переменные."
    )


if __name__ == "__main__":
    asyncio.run(import_environment())
