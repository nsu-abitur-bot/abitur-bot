"""Prepared popular-question snapshots. HTTP reads never run embeddings/clustering."""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any
from uuid import uuid4

from redis.asyncio import Redis
from redis.exceptions import RedisError, WatchError
from sqlalchemy.ext.asyncio import AsyncSession

from api.services.popular_questions import get_popular_questions
from config import instance
from config.instance import pinned_settings
from db.postgres.config import DATABASE_URL
from db.postgres.db import AsyncSessionLocal
from db.postgres.services.message_log import MessageLogService

logger = logging.getLogger(__name__)
DEFAULT_CONFIG = (500, 0.86)
SNAPSHOT_TTL = 24 * 60 * 60
REFRESH_INTERVAL = 15
REFRESH_TIMEOUT = 240
LOCK_TTL = 300
MAX_CONFIGS = 16


class PopularStatistics:
    def __init__(
        self,
        redis: Redis,
        session_factory: Callable[[], AsyncSession],
        *,
        embedder: Any | None = None,
        namespace: str | None = None,
    ) -> None:
        self.redis = redis
        self.session_factory = session_factory
        self.embedder = embedder
        identity = "|".join(
            [
                DATABASE_URL,
                os.getenv("INSTANCE_NAME", ""),
                os.getenv("CORS_ALLOW_ORIGINS", ""),
            ]
        )
        self.namespace = (
            namespace
            or hmac.new(
                os.getenv("SECRET_KEY", "").encode(), identity.encode(), hashlib.sha256
            ).hexdigest()[:24]
        )
        self.retry_after: dict[tuple[int, float], float] = {}
        self.configs: OrderedDict[tuple[int, float], None] = OrderedDict(
            [(DEFAULT_CONFIG, None)]
        )

    def _key(self, raw_limit: int, threshold: float) -> str:
        return (
            f"stats:popular:v2:{self.namespace}:{instance.revision()}:"
            f"{raw_limit}:{threshold.hex()}"
        )

    async def _payload(self, key: str) -> dict | None:
        raw = await self.redis.get(key)
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
            if not isinstance(payload.get("questions"), list):
                return None
            return payload
        except (ValueError, AttributeError):
            logger.warning("Invalid popular statistics snapshot")
            return None

    async def read(
        self, limit: int = 10, raw_limit: int = 500, similarity_threshold: float = 0.86
    ) -> dict:
        raw_limit = max(limit, raw_limit)
        config = (raw_limit, similarity_threshold)
        self.configs[config] = None
        self.configs.move_to_end(config)
        for old in list(self.configs):
            if len(self.configs) <= MAX_CONFIGS:
                break
            if old != DEFAULT_CONFIG:
                del self.configs[old]
                self.retry_after.pop(old, None)
        key = self._key(*config)
        payload = await self._payload(key)
        if payload is None:
            failed = bool(await self.redis.get(key + ":error"))
            return {
                "questions": [],
                "status": "error" if failed else "pending",
                "updated_at": None,
            }
        return {
            "questions": payload["questions"][:limit],
            "status": "ready",
            "updated_at": payload["updated_at"],
        }

    async def _unlock(self, key: str, owner: str) -> None:
        # WATCH prevents deleting a new owner's lease after ours has expired.
        async with self.redis.pipeline() as pipe:
            try:
                await pipe.watch(key)
                if await pipe.get(key) == owner:
                    pipe.multi()
                    pipe.delete(key)
                    await pipe.execute()
            except WatchError:
                pass

    async def _keep_lease(self, key: str, owner: str) -> None:
        while True:
            await asyncio.sleep(30)
            try:
                async with self.redis.pipeline() as pipe:
                    await pipe.watch(key)
                    if await pipe.get(key) != owner:
                        return
                    pipe.multi()
                    pipe.expire(key, LOCK_TTL)
                    await pipe.execute()
            except (RedisError, WatchError):
                logger.warning("Could not renew statistics refresh lease")

    @pinned_settings()
    async def refresh(self, raw_limit: int = 500, threshold: float = 0.86) -> None:
        key = self._key(raw_limit, threshold)
        owner = uuid4().hex
        if not await self.redis.set(key + ":lock", owner, nx=True, ex=LOCK_TTL):
            return
        keeper = asyncio.create_task(self._keep_lease(key + ":lock", owner))
        try:
            async with asyncio.timeout(REFRESH_TIMEOUT):
                async with self.session_factory() as session:
                    logs = MessageLogService(session)
                    groups = await logs.get_question_groups(raw_limit=raw_limit)
                    signature = hashlib.sha256(
                        json.dumps(groups, ensure_ascii=False, sort_keys=True).encode()
                    ).hexdigest()
                    existing = await self._payload(key)
                    if existing is not None and existing.get("signature") == signature:
                        await self.redis.expire(key, SNAPSHOT_TTL)
                        return
                    questions = await get_popular_questions(
                        logs,
                        limit=raw_limit,
                        raw_limit=raw_limit,
                        similarity_threshold=threshold,
                        embedder=self.embedder,
                        groups=groups,
                    )
                    # A changed model/configuration must not publish mixed revisions.
                    if key != self._key(raw_limit, threshold):
                        return
                    if await self.redis.get(key + ":lock") != owner:
                        return
                    payload = {
                        "questions": questions,
                        "updated_at": datetime.now(UTC).isoformat(),
                        "signature": signature,
                    }
                    await self.redis.set(
                        key, json.dumps(payload, ensure_ascii=False), ex=SNAPSHOT_TTL
                    )
                    await self.redis.delete(key + ":error")
        except asyncio.CancelledError:
            raise
        except Exception:
            await self.redis.set(key + ":error", "1", ex=120)
            raise
        finally:
            keeper.cancel()
            with suppress(asyncio.CancelledError):
                await keeper
            try:
                await self._unlock(key + ":lock", owner)
            except RedisError:
                logger.warning("Could not release statistics refresh lease")

    async def run(self) -> None:
        """Refresh changed question groups independently of page visits."""
        while True:
            configs = [DEFAULT_CONFIG, *(c for c in self.configs if c != DEFAULT_CONFIG)]
            for config in configs:
                if time.monotonic() < self.retry_after.get(config, 0):
                    continue
                try:
                    await self.refresh(*config)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Do not include provider errors: they can contain request contents.
                    logger.warning("Background popular statistics refresh failed")
                    self.retry_after[config] = time.monotonic() + 60
            await asyncio.sleep(REFRESH_INTERVAL)


@lru_cache(maxsize=1)
def get_popular_statistics() -> PopularStatistics:
    url = os.environ.get("REDIS_URL")
    if not url:
        raise RuntimeError("REDIS_URL is required for prepared statistics")
    redis = Redis.from_url(
        url, decode_responses=True, socket_timeout=2, socket_connect_timeout=2
    )
    return PopularStatistics(redis, AsyncSessionLocal)


async def run_popular_statistics() -> None:
    try:
        statistics = get_popular_statistics()
    except RuntimeError:
        logger.warning("Background statistics require REDIS_URL")
        return
    try:
        await statistics.run()
    finally:
        await statistics.redis.aclose()
