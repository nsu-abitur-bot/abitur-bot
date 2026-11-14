from datetime import datetime
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import AnswerFeedback, FeedbackReport, timestamp

RATINGS = {"positive", "negative"}


class AnswerFeedbackService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_candidate(
        self,
        user_id: int,
        session_id: str,
        channel: str,
        question: str,
        bot_response: str,
    ) -> AnswerFeedback:
        candidate = AnswerFeedback(
            user_id=user_id,
            session_id=session_id,
            channel=channel,
            question=question,
            bot_response=bot_response,
        )
        self.session.add(candidate)
        await self.session.commit()
        await self.session.refresh(candidate)
        return candidate

    async def rate(self, feedback_id: str, user_id: int, rating: str) -> bool:
        if rating not in RATINGS:
            raise ValueError("Invalid answer rating")
        result = await self.session.execute(
            update(AnswerFeedback)
            .where(
                AnswerFeedback.id == feedback_id,
                AnswerFeedback.user_id == user_id,
                AnswerFeedback.rating.is_(None),
            )
            .values(rating=rating, rated_at=timestamp())
            .returning(AnswerFeedback)
        )
        candidate = result.scalar_one_or_none()
        if candidate is None:
            return False

        # Снимок оцениваемого ответа важен: переписка могла уже продолжиться.
        self.session.add(
            FeedbackReport(
                user_id=candidate.user_id,
                session_id=candidate.session_id,
                channel=candidate.channel,
                comment="👍 Полезно" if rating == "positive" else "👎 Неверно",
                question=candidate.question,
                bot_response=candidate.bot_response,
                status="open",
            )
        )
        await self.session.commit()
        return True

    async def negative_by_question(
        self,
        start: Optional[datetime],
        end: Optional[datetime],
        limit: int = 10,
    ) -> list[dict]:
        """Вопросы с негативной оценкой ответов — «где бот отвечает плохо» (#262).

        Группировка по точному тексту вопроса: отчёт читает редактор, для
        него важен сам вопрос, а не канонизация формулировок.
        """
        negative_expr = func.count().filter(AnswerFeedback.rating == "negative")
        rated_expr = func.count().filter(AnswerFeedback.rating.isnot(None))
        stmt = (
            select(
                AnswerFeedback.question.label("question"),
                negative_expr.label("negative"),
                rated_expr.label("rated"),
            )
            .where(AnswerFeedback.rating.isnot(None))
            .group_by(AnswerFeedback.question)
            .having(negative_expr > 0)
            .order_by(negative_expr.desc())
            .limit(limit)
        )
        if start is not None:
            stmt = stmt.where(AnswerFeedback.created_at >= start.replace(tzinfo=None))
        if end is not None:
            stmt = stmt.where(AnswerFeedback.created_at <= end.replace(tzinfo=None))
        rows = (await self.session.execute(stmt)).all()
        return [
            {
                "question": row._mapping["question"],
                "negative": int(row._mapping["negative"]),
                "rated": int(row._mapping["rated"]),
            }
            for row in rows
        ]
