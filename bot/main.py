import asyncio
import logging
from os import getenv
from typing import Any, Dict

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv

from db.postgres.db import AsyncSessionLocal
from db.postgres.services.user import UserService
from llm.llm_client import ask_local_llm, cleanup_redis

load_dotenv()
BOT_TOKEN = getenv("BOT_TOKEN")
if BOT_TOKEN is None:
    raise ValueError("BOT_TOKEN не задан")

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))
dp = Dispatcher()

# Простое хранилище состояния сессий в памяти.
# Ключ — session_id (чат или chat:user для групп)
# Каждое значение — словарь с опциональными ключами:
# - 'awaiting_snils' (bool), 'snils' (str), 'history' (list)
# TODO: заменить на БД для долговременного хранения
session_states: Dict[str, Dict[str, Any]] = {}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("llm.llm_client").setLevel(logging.DEBUG)
logging.getLogger("db.redis_client").setLevel(logging.DEBUG)


def get_session_id(message: Message) -> str:
    """Вычисляет session_id для сообщения.

    В групповых чатах возвращает chat_id:user_id для изоляции пользователей.
    В личных чатах возвращает только chat_id.

    Raises:
        ValueError: Если message.from_user отсутствует.
    """
    if not message.from_user:
        raise ValueError("message.from_user is required")
    chat_id = str(message.chat.id)
    user_id = str(message.from_user.id)
    return (
        f"{chat_id}:{user_id}"
        if message.chat.type in ["group", "supergroup"]
        else chat_id
    )


@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Обработчик /start: приветствие и подсказка по командам."""
    if not message.from_user:
        return
    chat_id = str(message.chat.id)
    user_id = message.from_user.id

    async with AsyncSessionLocal() as session:
        user_service = UserService(session)
        existing_user = await user_service.get_user(user_id)

        if not existing_user:
            await user_service.create_user(user_id)
            logger.info(f"Новый пользователь {user_id} создан в БД")
        else:
            logger.info(f"Пользователь {user_id} уже существует в БД")

    await bot.send_message(chat_id, "Привет! Используйте /track, /untrack и /reset")


@dp.message(Command("track"))
async def cmd_track(message: Message):
    """Обработчик /track: переводим сессию в ожидание СНИЛС."""
    if not message.from_user:
        return
    chat_id = str(message.chat.id)
    session_id = get_session_id(message)
    state = session_states.setdefault(session_id, {})
    state["awaiting_snils"] = True
    await bot.send_message(chat_id, "Укажите свой СНИЛС")


@dp.message(Command("untrack"))
async def cmd_untrack(message: Message):
    """Обработчик /untrack: удаляем состояние сессии и СНИЛС из БД."""
    if not message.from_user:
        return
    chat_id = str(message.chat.id)
    user_id = message.from_user.id
    session_id = get_session_id(message)

    async with AsyncSessionLocal() as session:
        user_service = UserService(session)
        updated = await user_service.update_snils(user_id, None)
        if updated:
            logger.info(f"СНИЛС пользователя {user_id} удален из БД")
        else:
            logger.warning(f"Пользователь {user_id} не найден в БД для удаления СНИЛС")

    session_states.pop(session_id, None)
    await bot.send_message(chat_id, "Отслеживание прекращено")


@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    """Обработчик /reset: очищаем историю переписки для сессии."""
    if not message.from_user:
        return
    chat_id = str(message.chat.id)
    session_id = get_session_id(message)
    if session_id in session_states:
        session_states[session_id]["history"] = []
        await bot.send_message(chat_id, "История переписки очищена")
    else:
        await bot.send_message(chat_id, "Нет активной сессии для очистки")


@dp.message()
async def handle_message(message: Message):
    """Обычные сообщения: сохраняем СНИЛС или шлём в LLM."""
    if not message.from_user:
        return

    user_text = (message.text or "").strip()

    chat_id = str(message.chat.id)
    session_id = get_session_id(message)
    user_name = message.from_user.username or message.from_user.first_name or "Аноним"
    formatted_message = f"[from {user_name}] {user_text}"

    logger.info(f"Сообщение от {user_name} в чате {chat_id}: {user_text}")

    state = session_states.setdefault(session_id, {})

    # Если ожидали СНИЛС — сохраняем его
    if state.get("awaiting_snils"):
        async with AsyncSessionLocal() as session:
            user_service = UserService(session)
            user_id = message.from_user.id

            updated = await user_service.update_snils(user_id, user_text)
            if updated:
                logger.info(f"СНИЛС пользователя {user_id} обновлен в БД: {user_text}")
                state["snils"] = user_text
                state["awaiting_snils"] = False
                await bot.send_message(chat_id, "СНИЛС записан")
            else:
                logger.error(f"Не удалось обновить СНИЛС для пользователя {user_id}")
                await bot.send_message(
                    chat_id,
                    "Ошибка при сохранении СНИЛС.",
                )
        return

    # Пересылаем сообщение локальной модели LLM
    await bot.send_chat_action(chat_id, "typing")
    response = await ask_local_llm(formatted_message, session_id=session_id)

    if response:
        await bot.send_message(chat_id, response)


async def main():
    logger.info("Бот запущен")
    try:
        await dp.start_polling(bot)
    finally:
        # Закрываем Redis при завершении бота
        logger.info("Закрытие соединений...")
        await cleanup_redis()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен")
