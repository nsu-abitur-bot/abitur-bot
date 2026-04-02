"""Сервис для работы с категориями сообщений."""

import logging
from datetime import UTC, datetime
from typing import Optional

from config.question_categories import (
    ALL_CATEGORIES,
    NSU_RELATED_CATEGORIES,
    QuestionCategory,
    get_category_prompt,
    is_nsu_related,
)
from db.postgres.models import Message
from llm.factory import get_llm_provider
from sqlalchemy import select, text, func
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class MessageCategoryService:
    """Сервис для классификации и работы с категориями сообщений."""

    def __init__(self, db_session):
        self.session = db_session

    async def classify_message(self, user_text: str) -> Optional[str]:
        """
        Классифицирует сообщение с помощью LLM.
        
        Args:
            user_text: Текст сообщения пользователя
            
        Returns:
            Категория сообщения или None в случае ошибки
        """
        try:
            # Формируем промпт для классификации
            system_prompt = get_category_prompt()
            
            messages: list[BaseMessage] = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_text),
            ]
            
            # Получаем ответ от LLM
            provider = get_llm_provider()
            response = await provider.generate(messages)
            
            # Очищаем ответ и получаем категорию
            category = response.strip().lower()
            
            # Валидируем категорию
            if category not in ALL_CATEGORIES:
                logger.warning(f"Invalid category '{category}', falling back to 'general'")
                return QuestionCategory.GENERAL.value
                
            logger.info(f"Message classified as: {category}")
            return category
            
        except Exception as e:
            logger.error(f"Error classifying message: {e}")
            return QuestionCategory.GENERAL.value

    async def update_message_category(
        self, 
        message_id: str, 
        category: str,
        force_update: bool = False
    ) -> bool:
        """
        Обновляет категорию и дату тегирования сообщения.
        
        Args:
            message_id: ID сообщения
            category: Новая категория
            force_update: Принудительно обновить даже если уже есть категория
            
        Returns:
            True если обновлено успешно
        """
        try:
            # Валидируем категорию
            if category not in ALL_CATEGORIES:
                logger.error(f"Invalid category: {category}")
                return False
            
            # Находим сообщение
            message = await self.db_session.get(Message, message_id)
            if not message:
                logger.error(f"Message not found: {message_id}")
                return False
            
            # Проверяем нужно ли обновлять
            if message.category and not force_update:
                logger.info(f"Message {message_id} already has category: {message.category}")
                return True  # Считаем успешным, т.к. категория уже есть
            
            # Обновляем категорию и дату тегирования
            message.category = category
            message.tagged_at = datetime.now(UTC).replace(tzinfo=None)
            
            await self.db_session.commit()
            logger.info(f"Message {message_id} categorized as: {category}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating message category: {e}")
            await self.db_session.rollback()
            return False

    async def get_message_category(self, message_id: str) -> Optional[str]:
        """Получает категорию сообщения."""
        try:
            message = await self.db_session.get(Message, message_id)
            return message.category if message else None
        except Exception as e:
            logger.error(f"Error getting message category: {e}")
            return None

    async def get_messages_by_category(
        self, 
        category: str,
        limit: int = 100
    ) -> list[Message]:
        """Получает сообщения по категории."""
        try:
            if category not in ALL_CATEGORIES:
                logger.error(f"Invalid category: {category}")
                return []
            
            result = await self.session.execute(
                select(Message)
                .where(Message.category == category)
                .order_by(Message.created_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting messages by category: {e}")
            return []

    async def get_untagged_messages(self, limit: int = 1000) -> list[Message]:
        """Получает сообщения без категории для тегирования."""
        try:
            result = await self.session.execute(
                select(Message)
                .where(Message.category.is_(None))
                .order_by(Message.created_at.asc())
                .limit(limit)
            )
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting untagged messages: {e}")
            return []

    async def get_category_statistics(self) -> dict[str, int]:
        """Получает статистику по категориям."""
        try:
            stats = {}
            for category in ALL_CATEGORIES:
                result = await self.session.execute(
                    select(func.count(Message.id))
                    .where(Message.category == category)
                )
                count = result.scalar()
                stats[category] = count or 0
            
            # Добавляем общее количество и количество без категории
            result = await self.session.execute(
                select(func.count(Message.id))
            )
            total = result.scalar()
            stats["total"] = total or 0
            
            result = await self.session.execute(
                select(func.count(Message.id))
                .where(Message.category.is_(None))
            )
            untagged = result.scalar()
            stats["untagged"] = untagged or 0
            
            return stats
        except Exception as e:
            logger.error(f"Error getting category statistics: {e}")
            return {}

    async def batch_categorize_messages(self, limit: int = 100) -> int:
        """
        Массово категоризирует сообщения без категории.
        
        Returns:
            Количество обработанных сообщений
        """
        try:
            # Получаем сообщения без категории
            untagged_messages = await self.get_untagged_messages(limit)
            processed = 0
            
            for message in untagged_messages:
                # Классифицируем сообщение
                category = await self.classify_message(message.user_text)
                if category:
                    # Обновляем категорию
                    success = await self.update_message_category(
                        message.id, category, force_update=False
                    )
                    if success:
                        processed += 1
            
            logger.info(f"Batch categorized {processed} messages")
            return processed
            
        except Exception as e:
            logger.error(f"Error in batch categorization: {e}")
            return 0

    async def recategorize_by_date_range(
        self, 
        start_date: datetime,
        end_date: datetime,
        limit: int = 500
    ) -> int:
        """
        Перекатегоризирует сообщения в указанном диапазоне дат.
        Полезно при обновлении системы категоризации.
        
        Returns:
            Количество обработанных сообщений
        """
        try:
            result = await self.session.execute(
                select(Message)
                .where(Message.created_at.between(start_date, end_date))
                .order_by(Message.created_at.asc())
                .limit(limit)
            )
            messages = list(result.scalars().all())
            processed = 0
            
            for message in messages:
                # Классифицируем заново
                category = await self.classify_message(message.user_text)
                if category:
                    # Обновляем категорию с принудительным обновлением
                    success = await self.update_message_category(
                        message.id, category, force_update=True
                    )
                    if success:
                        processed += 1
            
            logger.info(f"Recategorized {processed} messages from {start_date} to {end_date}")
            return processed
            
        except Exception as e:
            logger.error(f"Error in recategorization: {e}")
            return 0
