"""Утилита для массового тегирования сообщений."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Optional

from db.postgres.db import AsyncSessionLocal
from db.postgres.services.message_category import MessageCategoryService

logger = logging.getLogger(__name__)


async def tag_existing_messages(
    batch_size: int = 100,
    max_messages: Optional[int] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> int:
    """
    Массово тегирует существующие сообщения без категории.
    
    Args:
        batch_size: Размер пакета для обработки
        max_messages: Максимальное количество сообщений для обработки
        date_from: Начальная дата для фильтрации
        date_to: Конечная дата для фильтрации
        
    Returns:
        Количество обработанных сообщений
    """
    total_processed = 0
    
    try:
        async with AsyncSessionLocal() as db_session:
            category_service = MessageCategoryService(db_session)
            
            while True:
                # Получаем сообщения без категории
                untagged_messages = await category_service.get_untagged_messages(batch_size)
                
                if not untagged_messages:
                    logger.info("No more untagged messages found")
                    break
                
                # Фильтруем по датам если указаны
                if date_from or date_to:
                    filtered_messages = []
                    for msg in untagged_messages:
                        msg_date = msg.created_at
                        if date_from and msg_date < date_from:
                            continue
                        if date_to and msg_date > date_to:
                            continue
                        filtered_messages.append(msg)
                    untagged_messages = filtered_messages
                
                if not untagged_messages:
                    logger.info("No messages match date filter")
                    break
                
                # Обрабатываем пакет
                batch_processed = 0
                for message in untagged_messages:
                    # Проверяем лимит
                    if max_messages and total_processed >= max_messages:
                        logger.info(f"Reached max messages limit: {max_messages}")
                        return total_processed
                    
                    # Классифицируем сообщение
                    category = await category_service.classify_message(message.user_text)
                    if category:
                        success = await category_service.update_message_category(
                            message.id, category, force_update=False
                        )
                        if success:
                            batch_processed += 1
                            total_processed += 1
                
                logger.info(f"Processed batch of {batch_processed} messages. Total: {total_processed}")
                
                # Небольшая задержка чтобы не перегружать LLM
                await asyncio.sleep(0.1)
                
                # Если обработали меньше чем размер пакета, значит достигли конца
                if batch_processed < batch_size:
                    break
    
    except Exception as e:
        logger.error(f"Error in batch tagging: {e}")
    
    return total_processed


async def retag_messages_by_period(
    start_date: datetime,
    end_date: datetime,
    batch_size: int = 100,
) -> int:
    """
    Перетегирует сообщения за указанный период.
    Полезно при обновлении системы категоризации.
    
    Args:
        start_date: Начальная дата
        end_date: Конечная дата
        batch_size: Размер пакета
        
    Returns:
        Количество обработанных сообщений
    """
    try:
        async with AsyncSessionLocal() as db_session:
            category_service = MessageCategoryService(db_session)
            
            total_processed = 0
            current_start = start_date
            
            while current_start < end_date:
                current_end = min(current_start + timedelta(days=7), end_date)
                
                processed = await category_service.recategorize_by_date_range(
                    current_start, current_end, batch_size
                )
                
                total_processed += processed
                logger.info(f"Recategorized {processed} messages from {current_start} to {current_end}")
                
                current_start = current_end
                
                # Если ничего не обработали, двигаемся дальше
                if processed == 0:
                    current_start = current_end
            
            return total_processed
            
    except Exception as e:
        logger.error(f"Error in recategorization: {e}")
        return 0


async def get_tagging_statistics() -> dict:
    """Получает статистику по тегированию сообщений."""
    try:
        async with AsyncSessionLocal() as db_session:
            category_service = MessageCategoryService(db_session)
            stats = await category_service.get_category_statistics()
            return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return {}


async def main():
    """Основная функция для запуска тегирования."""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Starting message tagging utility")
    
    # Получаем статистику до
    stats_before = await get_tagging_statistics()
    logger.info(f"Statistics before tagging: {stats_before}")
    
    # Тегируем сообщения за последнюю неделю
    week_ago = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=7)
    processed = await tag_existing_messages(
        batch_size=50,
        date_from=week_ago,
    )
    
    logger.info(f"Processed {processed} messages")
    
    # Получаем статистику после
    stats_after = await get_tagging_statistics()
    logger.info(f"Statistics after tagging: {stats_after}")
    
    # Показываем разницу
    if stats_before.get("untagged", 0) > 0:
        reduction = stats_before["untagged"] - stats_after.get("untagged", 0)
        logger.info(f"Reduced untagged messages by: {reduction}")


if __name__ == "__main__":
    asyncio.run(main())
