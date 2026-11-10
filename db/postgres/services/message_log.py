import hashlib
import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional, TypedDict

from sqlalchemy import Float, Sequence, func, or_
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from db.postgres.db import AsyncSessionLocal
from db.postgres.models import MessageLog, QuestionEmbeddingCache

logger = logging.getLogger(__name__)


class QuestionGroup(TypedDict):
    """Вопросы, совпадающие после нормализации текста."""

    canonical: str
    question: str
    count: int
    variants: list[str]


DEFAULT_QUESTION_RAW_LIMIT = 500
_FROM_PREFIX_RE = re.compile(r"^\[from\s+[^\]]*\]\s*", re.IGNORECASE)


def _question_text_without_author(text: str) -> str:
    return _FROM_PREFIX_RE.sub("", text).strip()


def _canonical_question(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text.strip(" \t\n\r.,!?;:()[]{}\"'")


def _question_hash(canonical_question: str) -> str:
    return hashlib.sha256(canonical_question.encode("utf-8")).hexdigest()


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
        stmt = (
            select(func.count())
            .select_from(MessageLog)
            .where(
                MessageLog.message_type == "user_input",
                MessageLog.created_at >= start,
            )
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

    async def get_faq_hit_stats(
        self,
        start: Optional[datetime],
        end: Optional[datetime],
        group_by: str,
    ) -> dict:
        """Как часто FAQ-слой отвечает без обращения к модели.

        Числитель — записи `faq_match`, знаменатель — `user_input`.
        Отдаётся по периодам, чтобы было видно динамику после смены порога.
        """
        if start is not None:
            start = start.replace(tzinfo=None)
        if end is not None:
            end = end.replace(tzinfo=None)

        period_expr = func.date_trunc(group_by, MessageLog.created_at).label("period")
        questions_expr = (
            func.count()
            .filter(MessageLog.message_type == "user_input")
            .label("questions")
        )
        hits_expr = (
            func.count().filter(MessageLog.message_type == "faq_match").label("hits")
        )
        stmt = (
            select(period_expr, questions_expr, hits_expr)
            .where(MessageLog.message_type.in_(["user_input", "faq_match"]))
            .group_by(period_expr)
            .order_by(period_expr)
        )
        if start is not None:
            stmt = stmt.where(MessageLog.created_at >= start)
        if end is not None:
            stmt = stmt.where(MessageLog.created_at <= end)

        rows = (await self.session.execute(stmt)).all()

        buckets = []
        total_questions = 0
        total_hits = 0
        for row in rows:
            if row.period is None:
                continue
            questions = int(row._mapping["questions"])
            hits = int(row._mapping["hits"])
            total_questions += questions
            total_hits += hits
            buckets.append(
                {
                    "period": row.period,
                    "questions": questions,
                    "hits": hits,
                    "hit_rate": hits / questions if questions else None,
                }
            )

        return {
            "total_questions": total_questions,
            "total_hits": total_hits,
            "hit_rate": total_hits / total_questions if total_questions else None,
            "buckets": buckets,
        }

    async def get_answer_stats(
        self,
        start: Optional[datetime],
        end: Optional[datetime],
        group_by: str,
    ) -> dict:
        """Наблюдаемость ответов по периодам (#242): latency, неответы, ошибки.

        Числитель метрик — записи `llm_response`, у которых пайплайн пишет
        `latency_ms` / `no_answer` / `error` в message_metadata; записи без
        этих ключей (старые данные) участвуют только в счётчике ответов.
        `questions` — все `user_input` периода: FAQ-попадания отвечают без
        `llm_response`, поэтому знаменатель доли неответов шире ответов модели.
        """
        if start is not None:
            start = start.replace(tzinfo=None)
        if end is not None:
            end = end.replace(tzinfo=None)

        # JSON-колонка приводится к jsonb ради ->>: так ключи читаются
        # текстом и кастуются на месте, NULL для отсутствующих ключей.
        meta = MessageLog.message_metadata.cast(JSONB)
        latency_expr = func.nullif(meta["latency_ms"].astext, "").cast(Float)
        no_answer_expr = meta["no_answer"].astext == "true"
        error_expr = meta["error"].astext == "true"

        period_expr = func.date_trunc(group_by, MessageLog.created_at).label("period")
        questions_expr = (
            func.count()
            .filter(MessageLog.message_type == "user_input")
            .label("questions")
        )
        answers_expr = (
            func.count()
            .filter(MessageLog.message_type == "llm_response")
            .label("answers")
        )
        no_answer_count_expr = func.count().filter(no_answer_expr).label("no_answer")
        error_count_expr = func.count().filter(error_expr).label("errors")
        p50_expr = func.percentile_cont(0.5).within_group(latency_expr).label("p50_ms")
        p95_expr = func.percentile_cont(0.95).within_group(latency_expr).label("p95_ms")

        stmt = (
            select(
                period_expr,
                questions_expr,
                answers_expr,
                no_answer_count_expr,
                error_count_expr,
                p50_expr,
                p95_expr,
            )
            .where(
                MessageLog.message_type.in_(["user_input", "llm_response"]),
            )
            .group_by(period_expr)
            .order_by(period_expr)
        )
        if start is not None:
            stmt = stmt.where(MessageLog.created_at >= start)
        if end is not None:
            stmt = stmt.where(MessageLog.created_at <= end)
        rows = (await self.session.execute(stmt)).all()

        def _bucket(mapping) -> dict:
            questions = int(mapping["questions"])
            answers = int(mapping["answers"])
            no_answer = int(mapping["no_answer"])
            errors = int(mapping["errors"])
            return {
                "period": mapping["period"],
                "questions": questions,
                "answers": answers,
                "no_answer": no_answer,
                "no_answer_share": no_answer / questions if questions else None,
                "errors": errors,
                "error_share": errors / questions if questions else None,
                "latency_ms_p50": mapping["p50_ms"],
                "latency_ms_p95": mapping["p95_ms"],
            }

        buckets = [
            bucket
            for row in rows
            if (bucket := _bucket(row._mapping))["period"] is not None
        ]
        totals_questions = sum(bucket["questions"] for bucket in buckets)
        totals_answers = sum(bucket["answers"] for bucket in buckets)
        totals_no_answer = sum(bucket["no_answer"] for bucket in buckets)
        totals_errors = sum(bucket["errors"] for bucket in buckets)
        return {
            "total_questions": totals_questions,
            "total_answers": totals_answers,
            "total_no_answer": totals_no_answer,
            "no_answer_share": (
                totals_no_answer / totals_questions if totals_questions else None
            ),
            "total_errors": totals_errors,
            "error_share": totals_errors / totals_questions if totals_questions else None,
            "buckets": buckets,
        }

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

        base_stmt = select(func.coalesce(func.sum(MessageLog.tokens_used), 0)).where(
            MessageLog.tokens_used.is_not(None)
        )

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
            buckets = [{"period": k, "tokens": v} for k, v in sorted(all_buckets.items())]
        else:
            buckets = []

        return {"total": total, "buckets": buckets}

    async def get_question_groups(
        self,
        raw_limit: int = DEFAULT_QUESTION_RAW_LIMIT,
        message_type: str = "user_input",
    ) -> List[QuestionGroup]:
        """Вопросы пользователей, сгруппированные по точному совпадению текста.

        `message_type` отделяет очередь неотвеченных (`no_answer`, #258) от
        всех вопросов. Объединение близких по смыслу формулировок делает
        `api/services/popular_questions.py`: для него нужны эмбеддинги, то есть
        вызов модели, и слою данных он не принадлежит.
        """
        stmt = (
            select(
                MessageLog.content.label("question"),
                func.count(MessageLog.id).label("count"),
            )
            .where(MessageLog.message_type == message_type)
            .group_by(MessageLog.content)
            .order_by(func.count(MessageLog.id).desc())
            .limit(raw_limit)
        )
        rows = (await self.session.execute(stmt)).all()

        groups: dict[str, QuestionGroup] = {}
        best_counts: dict[str, int] = {}
        for row in rows:
            question = row._mapping["question"]
            if question is None:
                continue

            question_text = _question_text_without_author(str(question))
            if not question_text:
                continue

            canonical = _canonical_question(question_text)
            count = int(row._mapping["count"])
            existing = groups.get(canonical)
            if existing is None:
                groups[canonical] = {
                    "canonical": canonical,
                    "question": question_text,
                    "count": count,
                    "variants": [question_text],
                }
                best_counts[canonical] = count
                continue

            existing["count"] += count
            # Представителем группы делаем самую частую формулировку.
            if count > best_counts[canonical]:
                existing["question"] = question_text
                best_counts[canonical] = count
            if question_text not in existing["variants"]:
                existing["variants"].append(question_text)

        return sorted(groups.values(), key=lambda item: item["count"], reverse=True)

    async def get_cached_question_vectors(
        self,
        provider: str,
        model: str,
        canonical_questions: list[str],
    ) -> dict[str, list[float]]:
        """Кэшированные векторы по каноническому тексту вопроса."""
        if not canonical_questions:
            return {}

        hash_to_canonical = {
            _question_hash(canonical): canonical for canonical in canonical_questions
        }
        stmt = select(QuestionEmbeddingCache).where(
            QuestionEmbeddingCache.provider == provider,
            QuestionEmbeddingCache.model == model,
            QuestionEmbeddingCache.question_hash.in_(list(hash_to_canonical)),
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return {
            hash_to_canonical[row.question_hash]: [
                float(value) for value in row.embedding
            ]
            for row in rows
            if isinstance(row.embedding, list) and row.question_hash in hash_to_canonical
        }

    async def store_question_vectors(
        self,
        provider: str,
        model: str,
        items: list[tuple[str, list[float]]],
    ) -> None:
        """Кладёт в кэш векторы: список пар (канонический текст, вектор)."""
        if not items:
            return

        for canonical, vector in items:
            self.session.add(
                QuestionEmbeddingCache(
                    provider=provider,
                    model=model,
                    question_hash=_question_hash(canonical),
                    canonical_question=canonical,
                    embedding=[float(value) for value in vector],
                )
            )

        try:
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            logger.warning("Could not store popular question embeddings cache")
