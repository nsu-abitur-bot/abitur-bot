"""Команды для управления категориями сообщений."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Optional

from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from config.admin import is_admin
from db.postgres.db import AsyncSessionLocal
from db.postgres.services.message_category import MessageCategoryService
from db.postgres.services.user import UserService

logger = logging.getLogger(__name__)


async def cmd_category_stats(message: Message):
    """Показать статистику по категориям."""
    if not is_admin(message.from_user.id):
        return
    
    try:
        async with AsyncSessionLocal() as db_session:
            category_service = MessageCategoryService(db_session)
            stats = await category_service.get_category_statistics()
            
            total = stats.get("total", 0)
            untagged = stats.get("untagged", 0)
            tagged = total - untagged
            
            stats_text = (
                f"📊 <b>Статистика категорий</b>\n\n"
                f"Всего сообщений: {total}\n"
                f"С категорией: {tagged}\n"
                f"Без категории: {untagged}\n"
                f"Процент тегированных: {(tagged/total*100):.1f}%\n\n"
                f"<b>По категориям:</b>\n"
            )
            
            # Добавляем статистику по каждой категории
            from config.question_categories import QuestionCategory, CATEGORY_DESCRIPTIONS
            
            for category in QuestionCategory:
                count = stats.get(category.value, 0)
                if count > 0:
                    percentage = (count / total * 100) if total > 0 else 0
                    stats_text += f"• {category.value}: {count} ({percentage:.1f}%)\n"
            
            await message.answer(stats_text, parse_mode="HTML")
            
    except Exception as e:
        logger.error(f"Error in category stats: {e}")
        await message.answer("Ошибка при получении статистики")


async def cmd_tag_messages(message: Message):
    """Запустить тегирование сообщений без категории."""
    if not is_admin(message.from_user.id):
        return
    
    try:
        await message.answer("🔄 Начинаю тегирование сообщений...")
        
        async with AsyncSessionLocal() as db_session:
            category_service = MessageCategoryService(db_session)
            
            # Тегируем последние 100 сообщений без категории
            processed = await category_service.batch_categorize_messages(limit=100)
            
            await message.answer(
                f"✅ Затегировано {processed} сообщений",
                parse_mode="HTML"
            )
            
    except Exception as e:
        logger.error(f"Error in tag messages: {e}")
        await message.answer("Ошибка при тегировании сообщений")


async def cmd_retag_week(message: Message):
    """Перетегировать сообщения за последнюю неделю."""
    if not is_admin(message.from_user.id):
        return
    
    try:
        await message.answer("🔄 Перетегирую сообщения за последнюю неделю...")
        
        week_ago = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=7)
        now = datetime.now(UTC).replace(tzinfo=None)
        
        async with AsyncSessionLocal() as db_session:
            category_service = MessageCategoryService(db_session)
            processed = await category_service.recategorize_by_date_range(
                week_ago, now, limit=500
            )
            
            await message.answer(
                f"✅ Перетегировано {processed} сообщений за последнюю неделю",
                parse_mode="HTML"
            )
            
    except Exception as e:
        logger.error(f"Error in retag week: {e}")
        await message.answer("Ошибка при перетегировании")


async def cmd_category_info(message: Message):
    """Показать информацию о категориях."""
    if not is_admin(message.from_user.id):
        return
    
    try:
        from config.question_categories import QuestionCategory, CATEGORY_DESCRIPTIONS
        
        info_text = "<b>📋 Доступные категории:</b>\n\n"
        
        for category in QuestionCategory:
            description = CATEGORY_DESCRIPTIONS[category]
            info_text += f"• <code>{category.value}</code> - {description}\n"
        
        info_text += (
            "\n<b>Использование:</b>\n"
            "Каждое сообщение автоматически классифицируется "
            "в одну из этих категорий при обработке."
        )
        
        await message.answer(info_text, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in category info: {e}")
        await message.answer("Ошибка при получении информации о категориях")


def register_category_handlers(dp: Dispatcher):
    """Регистрирует обработчики для управления категориями."""
    dp.message.register(cmd_category_stats, Command("category_stats"))
    dp.message.register(cmd_tag_messages, Command("tag_messages"))
    dp.message.register(cmd_retag_week, Command("retag_week"))
    dp.message.register(cmd_category_info, Command("category_info"))
