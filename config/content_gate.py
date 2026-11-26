"""Cross-process fence for index/FAQ writers during a model migration."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import IO, AsyncGenerator

from config.instance import generation_id


class ModelMigrationBusy(ValueError):
    pass


def storage_root() -> Path:
    return Path(os.getenv("LIGHTRAG_WORKSPACE_BASE", "./data/lightrag"))


def migration_marker() -> Path:
    return storage_root() / ".model-migration.json"


def active_migration() -> str | None:
    path = migration_marker()
    return json.loads(path.read_text())["id"] if path.exists() else None


@asynccontextmanager
async def index_fence(*, exclusive: bool = False) -> AsyncGenerator[IO[str] | None, None]:
    if os.name != "posix":
        if exclusive or active_migration():
            raise ModelMigrationBusy("Пересчёт индексов поддерживается на Linux/macOS")
        yield None
        return
    import fcntl

    root = storage_root()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".model-write.lock").open("a+") as lock:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        while True:
            try:
                fcntl.flock(lock, mode | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not exclusive:
                    raise ModelMigrationBusy(
                        "Идёт переключение индекса; повторите правку"
                    ) from None
                await asyncio.sleep(0.05)
        try:
            yield lock
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@asynccontextmanager
async def content_write() -> AsyncGenerator[None, None]:
    async with index_fence():
        if active_migration():
            raise ModelMigrationBusy(
                "Идёт пересчёт эмбеддингов. Правки материалов и FAQ временно "
                "недоступны; поиск продолжает работать"
            )
        published = storage_root() / ".model-generation.json"
        if (
            published.exists()
            and json.loads(published.read_text())["id"] != generation_id()
        ):
            raise ModelMigrationBusy(
                "Настройки индекса обновляются; повторите правку через несколько секунд"
            )
        yield
