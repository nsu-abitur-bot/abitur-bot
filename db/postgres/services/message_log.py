import asyncio
import hashlib
import logging
import math
import os
import re
from datetime import datetime, timedelta
from typing import Any, List, Optional, TypedDict

from sqlalchemy import Sequence, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from db.postgres.db import AsyncSessionLocal
from db.postgres.models import MessageLog, QuestionEmbeddingCache
from llm.factory import get_llm_provider

logger = logging.getLogger(__name__)

DEFAULT_POPULAR_QUESTION_SIMILARITY_THRESHOLD = 0.86
DEFAULT_POPULAR_QUESTION_RAW_LIMIT = 500
MAX_POPULAR_QUESTION_VARIANTS = 5
_FROM_PREFIX_RE = re.compile(r"^\[from\s+[^\]]*\]\s*", re.IGNORECASE)


def _question_text_without_author(text: str) -> str:
    return _FROM_PREFIX_RE.sub("", text).strip()


def _canonical_question(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text.strip(" \t\n\r.,!?;:()[]{}\"'")


def _question_hash(canonical_question: str) -> str:
    return hashlib.sha256(canonical_question.encode("utf-8")).hexdigest()


def _embedder_cache_identity(embedder: Any) -> tuple[str, str]:
    module_name = embedder.__class__.__module__.lower()
    class_name = embedder.__class__.__name__.lower()
    if "openai" in module_name or "openai" in class_name:
        provider = "openai"
    elif "gemini" in module_name or "gemini" in class_name:
        provider = "gemini"
    else:
        provider = os.getenv("LLM_PROVIDER", "") or class_name

    model = (
        getattr(embedder, "model", None)
        or getattr(embedder, "model_name", None)
        or getattr(embedder, "model_id", None)
        or "default"
    )
    return str(provider).lower(), str(model)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _truncate_dt(dt: datetime, group_by: str) -> datetime:
    if group_by == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    if group_by == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if group_by == "week":
        return (dt - timedelta(days=dt.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    if group_by == "month":
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return dt


def _advance_dt(dt: datetime, group_by: str) -> datetime:
    if group_by == "hour":
        return dt + timedelta(hours=1)
    if group_by == "day":
        return dt + timedelta(days=1)
    if group_by == "week":
        return dt + timedelta(weeks=1)
    if group_by == "month":
        month = dt.month % 12 + 1
        year = dt.year + (1 if dt.month == 12 else 0)
        return dt.replace(year=year, month=month)
    return dt


class MessageLogService:
    """Сервис для работы с логами сообщений."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_log(
        self,
        user_id: int,
        session_id: str,
        message_type: str,
        content: str,
        message_metadata: Optional[dict] = None,
        topic_id: Optional[int] = None,
        tokens_used: Optional[int] = None,
    ) -> MessageLog:
        """Создает запись в логе сообщений."""
        log_entry = MessageLog(
            user_id=user_id,
            session_id=session_id,
            message_type=message_type,
            content=content,
            message_metadata=message_metadata,
            topic_id=topic_id,
            tokens_used=tokens_used,
        )
        self.session.add(log_entry)
        await self.session.commit()
        await self.session.refresh(log_entry)
        return log_entry

    async def count_user_inputs_since(
        self,
        start: datetime,
        user_id: Optional[int] = None,
    ) -> int:
        """Считает пользовательские запросы с указанного времени."""
        stmt = select(func.count()).select_from(MessageLog).where(
            MessageLog.message_type == "user_input",
            MessageLog.created_at >= start,
        )
        if user_id is not None:
            stmt = stmt.where(MessageLog.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def get_logs_by_session(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[MessageLog]:
        """Получает логи для конкретной сессии."""
        session_filter = MessageLog.session_id == session_id
        if ":dialog:" not in session_id:
            session_filter = or_(
                session_filter,
                MessageLog.session_id.like(f"{session_id}:dialog:%"),
            )

        stmt = (
            select(MessageLog)
            .where(session_filter)
            .order_by(MessageLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_logs_by_user(
        self,
        user_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[MessageLog]:
        """Получает логи для конкретного пользователя."""
        stmt = (
            select(MessageLog)
            .where(MessageLog.user_id == user_id)
            .order_by(MessageLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update_log_topic(self, log_id: int, topic_id: Optional[int]) -> bool:
        """Обновляет topic_id для лога."""
        try:
            async with AsyncSessionLocal() as session:
                stmt = select(MessageLog).where(MessageLog.id == log_id)
                result = await session.execute(stmt)
                log_entry = result.scalar_one_or_none()
                if log_entry:
                    log_entry.topic_id = topic_id
                    await session.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Ошибка обновления topic_id в логе: {e}")
            return False

    async def get_logs_by_type(
        self, message_type: str, limit: int = 100, offset: int = 0
    ) -> Sequence[MessageLog]:
        """Получает логи по типу сообщения."""
        stmt = (
            select(MessageLog)
            .where(MessageLog.message_type == message_type)
            .limit(limit)
            .offset(offset)
        )
        return stmt

    async def get_recent_logs(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[MessageLog]:
        """Получает самые последние логи."""
        stmt = (
            select(MessageLog)
            .order_by(MessageLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_request_count_stats(
        self,
        start: Optional[datetime],
        end: Optional[datetime],
        group_by: str,
        message_type: str = "user_input",
    ) -> dict:
        """Возвращает статистику количества запросов по периодам."""
        if start is not None:
            start = start.replace(tzinfo=None)
        if end is not None:
            end = end.replace(tzinfo=None)

        base_stmt = select(func.count(MessageLog.id)).where(
            MessageLog.message_type == message_type
        )

        if start is not None:
            base_stmt = base_stmt.where(MessageLog.created_at >= start)
        if end is not None:
            base_stmt = base_stmt.where(MessageLog.created_at <= end)

        total = (await self.session.execute(base_stmt)).scalar_one()

        period_expr = func.date_trunc(group_by, MessageLog.created_at).label("period")
        bucket_stmt = select(period_expr, func.count(MessageLog.id).label("count")).where(
            MessageLog.message_type == message_type
        )
        if start is not None:
            bucket_stmt = bucket_stmt.where(MessageLog.created_at >= start)
        if end is not None:
            bucket_stmt = bucket_stmt.where(MessageLog.created_at <= end)

        bucket_stmt = bucket_stmt.group_by(period_expr).order_by(period_expr)
        rows = (await self.session.execute(bucket_stmt)).all()

        valid_rows = [row for row in rows if row.period is not None]
        if valid_rows:
            range_start = (
                _truncate_dt(start, group_by)
                if start is not None
                else min(row.period for row in valid_rows)
            )
            range_end = end if end is not None else max(row.period for row in valid_rows)
            all_buckets: dict[datetime, int] = {}
            current = range_start
            while current <= range_end:
                all_buckets[current] = 0
                current = _advance_dt(current, group_by)
            for row in valid_rows:
                if row.period in all_buckets:
                    count_value = row._mapping["count"]
                    all_buckets[row.period] = int(count_value)
            buckets = [{"period": k, "count": v} for k, v in sorted(all_buckets.items())]
        else:
            buckets = []

        return {"total": total, "buckets": buckets}

    async def get_token_usage_stats(
        self,
        start: Optional[datetime],
        end: Optional[datetime],
        group_by: str,
    ) -> dict:
        """Возвращает статистику суммарного потребления токенов по периодам.

        Считаются только записи, у которых заполнено `tokens_used`
        (то есть `llm_response`).
        """
        if start is not None:
            start = start.replace(tzinfo=None)
        if end is not None:
            end = end.replace(tzinfo=None)

        base_stmt = select(
            func.coalesce(func.sum(MessageLog.tokens_used), 0)
        ).where(MessageLog.tokens_used.is_not(None))

        if start is not None:
            base_stmt = base_stmt.where(MessageLog.created_at >= start)
        if end is not None:
            base_stmt = base_stmt.where(MessageLog.created_at <= end)

        total = int((await self.session.execute(base_stmt)).scalar_one() or 0)

        period_expr = func.date_trunc(group_by, MessageLog.created_at).label("period")
        bucket_stmt = select(
            period_expr,
            func.coalesce(func.sum(MessageLog.tokens_used), 0).label("tokens"),
        ).where(MessageLog.tokens_used.is_not(None))

        if start is not None:
            bucket_stmt = bucket_stmt.where(MessageLog.created_at >= start)
        if end is not None:
            bucket_stmt = bucket_stmt.where(MessageLog.created_at <= end)

        bucket_stmt = bucket_stmt.group_by(period_expr).order_by(period_expr)
        rows = (await self.session.execute(bucket_stmt)).all()

        valid_rows = [row for row in rows if row.period is not None]
        if valid_rows:
            range_start = (
                _truncate_dt(start, group_by)
                if start is not None
                else min(row.period for row in valid_rows)
            )
            range_end = end if end is not None else max(row.period for row in valid_rows)
            all_buckets: dict[datetime, int] = {}
            current = range_start
            while current <= range_end:
                all_buckets[current] = 0
                current = _advance_dt(current, group_by)
            for row in valid_rows:
                if row.period in all_buckets:
                    all_buckets[row.period] = int(row._mapping["tokens"] or 0)
            buckets = [
                {"period": k, "tokens": v} for k, v in sorted(all_buckets.items())
            ]
        else:
            buckets = []

        return {"total": total, "buckets": buckets}

    class PopularQuestionRow(TypedDict):
        question: str
        count: int
        variants: list[str]

    async def _get_cached_question_vectors(
        self,
        provider: str,
        model: str,
        question_hashes: list[str],
    ) -> dict[str, list[float]]:
        if not question_hashes:
            return {}

        stmt = select(QuestionEmbeddingCache).where(
            QuestionEmbeddingCache.provider == provider,
            QuestionEmbeddingCache.model == model,
            QuestionEmbeddingCache.question_hash.in_(question_hashes),
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return {
            row.question_hash: [float(value) for value in row.embedding]
            for row in rows
            if isinstance(row.embedding, list)
        }

    async def _store_question_vectors(
        self,
        provider: str,
        model: str,
        questions: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> None:
        if not questions:
            return

        for item, vector in zip(questions, vectors):
            self.session.add(
                QuestionEmbeddingCache(
                    provider=provider,
                    model=model,
                    question_hash=_question_hash(item["canonical"]),
                    canonical_question=item["canonical"],
                    embedding=[float(value) for value in vector],
                )
            )

        try:
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            logger.warning("Could not store popular question embeddings cache")

    @staticmethod
    def _popular_question_rows(
        questions: list[dict[str, Any]],
        limit: int,
    ) -> List[PopularQuestionRow]:
        return [
            {
                "question": item["question"],
                "count": item["count"],
                "variants": item["variants"][:MAX_POPULAR_QUESTION_VARIANTS],
            }
            for item in questions[:limit]
        ]

    async def get_popular_questions(
        self,
        limit: int = 10,
        raw_limit: int = DEFAULT_POPULAR_QUESTION_RAW_LIMIT,
        similarity_threshold: float = DEFAULT_POPULAR_QUESTION_SIMILARITY_THRESHOLD,
        embedder: Any | None = None,
    ) -> List[PopularQuestionRow]:
        """Возвращает популярные вопросы с объединением похожих формулировок."""
        raw_limit = max(limit, raw_limit)
        stmt = (
            select(
                MessageLog.content.label("question"),
                func.count(MessageLog.id).label("count"),
            )
            .where(MessageLog.message_type == "user_input")
            .group_by(MessageLog.content)
            .order_by(func.count(MessageLog.id).desc())
            .limit(raw_limit)
        )
        rows = (await self.session.execute(stmt)).all()

        exact_groups: dict[str, MessageLogService.PopularQuestionRow] = {}
        exact_group_best_counts: dict[str, int] = {}
        for row in rows:
            question = row._mapping["question"]
            if question is None:
                continue

            question_text = _question_text_without_author(str(question))
            if not question_text:
                continue

            canonical = _canonical_question(question_text)
            count = int(row._mapping["count"])
            existing = exact_groups.get(canonical)
            if existing is None:
                exact_groups[canonical] = {
                    "question": question_text,
                    "count": count,
                    "variants": [question_text],
                }
                exact_group_best_counts[canonical] = count
                continue

            existing["count"] += count
            if count > exact_group_best_counts[canonical]:
                existing["question"] = question_text
                exact_group_best_counts[canonical] = count
            if question_text not in existing["variants"]:
                existing["variants"].append(question_text)

        questions: list[dict[str, Any]] = sorted(
            [
                {
                    "canonical": canonical,
                    "question": item["question"],
                    "count": item["count"],
                    "variants": item["variants"],
                }
                for canonical, item in exact_groups.items()
            ],
            key=lambda item: item["count"],
            reverse=True,
        )
        if not questions:
            return []

        if embedder is None:
            embedder = get_llm_provider().get_embeddings_model()
        if embedder is None:
            logger.warning("LLM provider does not expose embeddings; using exact groups")
            return self._popular_question_rows(questions, limit)

        provider, model = _embedder_cache_identity(embedder)
        hashes = [_question_hash(item["canonical"]) for item in questions]
        vectors_by_hash = await self._get_cached_question_vectors(
            provider,
            model,
            hashes,
        )
        missing_questions = [
            item
            for item, question_hash in zip(questions, hashes)
            if question_hash not in vectors_by_hash
        ]

        if missing_questions:
            missing_vectors = await asyncio.to_thread(
                embedder.embed_documents,
                [item["question"] for item in missing_questions],
            )
            if len(missing_vectors) != len(missing_questions):
                logger.warning(
                    "Embeddings count mismatch for popular questions: "
                    "%d vectors for %d questions",
                    len(missing_vectors),
                    len(missing_questions),
                )
                return self._popular_question_rows(questions, limit)

            normalized_vectors = [
                [float(value) for value in vector] for vector in missing_vectors
            ]
            for item, vector in zip(missing_questions, normalized_vectors):
                vectors_by_hash[_question_hash(item["canonical"])] = vector
            await self._store_question_vectors(
                provider,
                model,
                missing_questions,
                normalized_vectors,
            )

        vectors = [vectors_by_hash[question_hash] for question_hash in hashes]

        clusters: list[dict[str, Any]] = []
        for item, vector in zip(questions, vectors):
            vector = [float(value) for value in vector]
            best_cluster: dict[str, Any] | None = None
            best_similarity = similarity_threshold
            for cluster in clusters:
                similarity = _cosine_similarity(vector, cluster["centroid"])
                if similarity >= best_similarity:
                    best_cluster = cluster
                    best_similarity = similarity

            if best_cluster is None:
                clusters.append(
                    {
                        "question": item["question"],
                        "count": item["count"],
                        "variants": list(item["variants"]),
                        "centroid": vector,
                        "representative_count": item["count"],
                    }
                )
                continue

            old_count = best_cluster["count"]
            new_count = old_count + item["count"]
            best_cluster["centroid"] = [
                ((old_value * old_count) + (new_value * item["count"])) / new_count
                for old_value, new_value in zip(best_cluster["centroid"], vector)
            ]
            best_cluster["count"] = new_count
            if item["count"] > best_cluster["representative_count"]:
                best_cluster["question"] = item["question"]
                best_cluster["representative_count"] = item["count"]
            for variant in [item["question"], *item["variants"]]:
                if variant not in best_cluster["variants"]:
                    best_cluster["variants"].append(variant)

        clusters.sort(key=lambda item: item["count"], reverse=True)
        return [
            {
                "question": cluster["question"],
                "count": cluster["count"],
                "variants": cluster["variants"][:MAX_POPULAR_QUESTION_VARIANTS],
            }
            for cluster in clusters[:limit]
        ]
