import logging
from datetime import date, datetime, time, timedelta
from typing import List, Optional

from sqlalchemy import String, cast, func, or_, select, true
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import contains_eager, joinedload

from ..models import Message, User

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
                .options(joinedload(Message.user))
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
                .options(joinedload(Message.user))
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
                .options(joinedload(Message.user))
                .order_by(Message.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())
        except Exception as e:
            logger.error("Ошибка получения всех сообщений: %s", e)
            return []

    async def get_message_page(
        self,
        *,
        limit: int = 25,
        offset: int = 0,
        user_id: int | None = None,
        search: str | None = None,
        channel: str | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        snapshot_id: str | None = None,
    ) -> tuple[list[Message], int, str | None]:
        """Count and page share one PostgreSQL statement snapshot."""
        query = select(Message.id, Message.created_at).join(User)
        if snapshot_id is not None:
            query = query.where(Message.id <= snapshot_id)
        if user_id is not None:
            query = query.where(Message.user_id == user_id)
        if channel == "telegram":
            query = query.where(User.telegram_id.is_not(None))
        elif channel == "max":
            query = query.where(User.telegram_id.is_(None), User.max_id.is_not(None))
        if from_date is not None:
            query = query.where(
                Message.created_at >= datetime.combine(from_date, time.min)
            )
        if to_date is not None:
            # The DB stores naive UTC. A half-open bound includes fractional seconds.
            if to_date == date.max:
                query = query.where(Message.created_at <= datetime.max)
            else:
                query = query.where(
                    Message.created_at
                    < datetime.combine(to_date + timedelta(days=1), time.min)
                )
        if search and (term := search.strip()):
            pattern = (
                "%" + term.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%"
            )
            query = query.where(
                or_(
                    Message.user_text.ilike(pattern, escape="!"),
                    Message.bot_response.ilike(pattern, escape="!"),
                    Message.session_id.ilike(pattern, escape="!"),
                    cast(Message.user_id, String).ilike(pattern, escape="!"),
                )
            )
        filtered = query.cte("filtered_messages")
        page = (
            select(filtered)
            .order_by(filtered.c.created_at.desc(), filtered.c.id.desc())
            .limit(limit)
            .offset(offset)
            .cte("message_page")
        )
        total = (
            select(
                func.count().label("total"),
                func.max(filtered.c.id).label("snapshot_id"),
            )
            .select_from(filtered)
            .subquery()
        )
        result = await self.session.execute(
            select(Message, total.c.total, total.c.snapshot_id)
            .select_from(total.outerjoin(page, true()))
            .outerjoin(Message, Message.id == page.c.id)
            .outerjoin(User, User.user_id == Message.user_id)
            .options(contains_eager(Message.user))
            .order_by(Message.created_at.desc(), Message.id.desc())
        )
        rows = result.all()
        return (
            [row[0] for row in rows if row[0] is not None],
            rows[0][1],
            snapshot_id if snapshot_id is not None else rows[0][2],
        )
