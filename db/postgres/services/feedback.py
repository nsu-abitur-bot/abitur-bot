import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Feedback

logger = logging.getLogger(__name__)


class FeedbackService:
    """Сервис для сохранения обратной связи пользователей."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_feedback(
        self,
        user_id: int,
        session_id: str,
        text: str,
    ) -> Optional[Feedback]:
        """Сохраняет обратную связь пользователя в БД."""
        try:
            feedback = Feedback(user_id=user_id, session_id=session_id, text=text)
            self.session.add(feedback)
            await self.session.commit()
            await self.session.refresh(feedback)
            logger.info("Обратная связь сохранена для user_id=%s", user_id)
            return feedback
        except Exception as e:
            await self.session.rollback()
            logger.error("Ошибка сохранения обратной связи: %s", e)
            return None
