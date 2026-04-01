import logging
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Message

logger = logging.getLogger(__name__)


class MessageService:
    """Сервис для работы с сообщениями."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_message(
        self,
        user_id: int,
        session_id: str,
        user_text: str,
        bot_response: str,
    ) -> Optional[Message]:
        """Сохраняет пару (вопрос пользователя, ответ бота) в БД."""
        try:
            msg = Message(
                user_id=user_id,
                session_id=session_id,
                user_text=user_text,
                bot_response=bot_response,
            )
            self.session.add(msg)
            await self.session.commit()
            await self.session.refresh(msg)
            logger.info("Сообщение сохранено для user_id=%s", user_id)
            return msg
        except Exception as e:
            await self.session.rollback()
            logger.error("Ошибка сохранения сообщения: %s", e)
            return None

    async def get_messages_by_user(
        self, user_id: int, limit: int = 50, offset: int = 0
    ) -> List[Message]:
        """Получает сообщения пользователя с пагинацией."""
        try:
            result = await self.session.execute(
                select(Message)
                .where(Message.user_id == user_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())
        except Exception as e:
            logger.error("Ошибка получения сообщений для user_id=%s: %s", user_id, e)
            return []

    async def get_messages_by_session(
        self, session_id: str, limit: int = 50, offset: int = 0
    ) -> List[Message]:
        """Получает сообщения по session_id с пагинацией."""
        try:
            result = await self.session.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())
        except Exception as e:
            logger.error(
                "Ошибка получения сообщений для session_id=%s: %s", session_id, e
            )
            return []

    async def get_all_messages(self, limit: int = 50, offset: int = 0) -> List[Message]:
        """Получает все сообщения с пагинацией."""
        try:
            result = await self.session.execute(
                select(Message)
                .order_by(Message.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())
        except Exception as e:
            logger.error("Ошибка получения всех сообщений: %s", e)
            return []
