from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from db.postgres.models import MessageLog


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
        metadata: Optional[dict] = None,
    ) -> MessageLog:
        """Создает запись в логе сообщений."""
        log_entry = MessageLog(
            user_id=user_id,
            session_id=session_id,
            message_type=message_type,
            content=content,
            metadata=metadata,
        )
        self.session.add(log_entry)
        await self.session.commit()
        await self.session.refresh(log_entry)
        return log_entry

    async def get_logs_by_session(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MessageLog]:
        """Получает логи для конкретной сессии."""
        stmt = (
            select(MessageLog)
            .where(MessageLog.session_id == session_id)
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
    ) -> List[MessageLog]:
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

    async def get_logs_by_type(
        self,
        message_type: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MessageLog]:
        """Получает логи по типу сообщения."""
        stmt = (
            select(MessageLog)
            .where(MessageLog.message_type == message_type)
            .order_by(MessageLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_recent_logs(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[MessageLog]:
        """Получает самые последние логи."""
        stmt = (
            select(MessageLog)
            .order_by(MessageLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
