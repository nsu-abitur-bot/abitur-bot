"""Отчёт о пробелах наполнения (#262).

Собирает в одном ответе то, что редактору иначе пришлось бы собирать из
четырёх экранов: какие темы спрашивают, что популярно, на чём бот честно
отвечает «не нашёл», где доля неответов/ошибок высока и какие ответы
пользователи оценили негативно. Всё — из уже накопленных логов и оценок,
без вызовов моделей: отчёт должен строиться дёшево и детерминированно.

Семантическое объединение близких формулировок (эмбеддинги) здесь не
применяется: точная группировка даёт достаточно сигнала для редактора,
а LLM-вызовы в GET-эндпоинте недопустимы.
"""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from db.postgres.models import MessageLog, Topic
from db.postgres.services.answer_feedback import AnswerFeedbackService
from db.postgres.services.message_log import MessageLogService, QuestionGroup

logger = logging.getLogger(__name__)


async def _topic_distribution(
    session: AsyncSession, start: datetime, end: datetime, limit: int
) -> list[dict]:
    """Какие темы спрашивают за период (та же связка, что /stats/topics)."""
    stmt = (
        select(Topic.label, func.count(MessageLog.id).label("count"))
        .join(MessageLog, Topic.id == MessageLog.topic_id, isouter=True)
        .where(
            Topic.is_active,
            MessageLog.message_type == "user_input",
            MessageLog.created_at >= start.replace(tzinfo=None),
            MessageLog.created_at <= end.replace(tzinfo=None),
        )
        .group_by(Topic.id, Topic.label)
        .order_by(func.count(MessageLog.id).desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [
        {"topic": row._mapping["label"], "count": int(row._mapping["count"])}
        for row in rows
    ]


async def build_gap_report(
    session: AsyncSession, days: int = 30, limit: int = 10
) -> dict:
    log_service = MessageLogService(session)
    feedback_service = AnswerFeedbackService(session)

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)

    # Топ без эмбеддингов: точные группы уже несут частоту и варианты.
    popular = await log_service.get_question_groups(
        raw_limit=500, message_type="user_input"
    )
    unanswered = await log_service.get_question_groups(
        raw_limit=500, message_type="no_answer"
    )
    stats = await log_service.get_answer_stats(start=start, end=None, group_by="day")

    def _top(groups: list[QuestionGroup]) -> list[dict]:
        return [
            {
                "question": group["question"],
                "count": group["count"],
                "variants": group["variants"][:5],
            }
            for group in groups[:limit]
        ]

    return {
        "period_days": days,
        "generated_at": now.isoformat(),
        "topics": await _topic_distribution(session, start, now, limit),
        "popular": _top(popular),
        "unanswered": _top(unanswered),
        "quality": {
            "questions": stats["total_questions"],
            "answers": stats["total_answers"],
            "no_answer": stats["total_no_answer"],
            "no_answer_share": stats["no_answer_share"],
            "errors": stats["total_errors"],
            "error_share": stats["error_share"],
        },
        "disliked": await feedback_service.negative_by_question(start, None, limit),
    }
