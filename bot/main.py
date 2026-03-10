import asyncio
import logging
import re
from os import getenv

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv

from db.postgres.db import AsyncSessionLocal
from db.postgres.services.user import UserService
from llm.llm_client import ask_local_llm, cleanup_redis, get_redis_client

load_dotenv()
BOT_TOKEN = getenv("BOT_TOKEN")
if BOT_TOKEN is None:
    raise ValueError("BOT_TOKEN не задан")

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))
dp = Dispatcher()


logger = logging.getLogger(__name__)


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
    session_id = get_session_id(message)

    async with AsyncSessionLocal() as session:
        user_service = UserService(session)
        existing_user = await user_service.get_user(user_id)

        if not existing_user:
            await user_service.create_user(user_id)
            logger.info(f"Новый пользователь {user_id} создан в БД")
        else:
            logger.info(f"Пользователь {user_id} уже существует в БД")

    # Сбрасываем флаг ожидания applicant_id на случай зависшего состояния
    redis_client = await get_redis_client()
    await redis_client.set_awaiting_applicant_id(session_id, False)

    await bot.send_message(chat_id, "Привет! Используйте /track, /untrack и /reset")


@dp.message(Command("track"))
async def cmd_track(message: Message):
    """Обработчик /track: переводим сессию в ожидание идентификатора абитуриента."""
    if not message.from_user:
        return
    chat_id = str(message.chat.id)
    session_id = get_session_id(message)

    redis_client = await get_redis_client()
    await redis_client.set_awaiting_applicant_id(session_id, True)

    await bot.send_message(chat_id, "Укажите свой идентификатор абитуриента")


@dp.message(Command("untrack"))
async def cmd_untrack(message: Message):
    """Обработчик /untrack: удаляем состояние сессии и идентификатор
    абитуриента из БД."""
    if not message.from_user:
        return
    chat_id = str(message.chat.id)
    user_id = message.from_user.id
    session_id = get_session_id(message)

    async with AsyncSessionLocal() as session:
        user_service = UserService(session)
        updated = await user_service.update_applicant_id(user_id, None)
        if updated:
            logger.info(f"Идентификатор пользователя {user_id} удален из БД")
        else:
            logger.warning(
                f"Пользователь {user_id} не найден в БД для удаления идентификатора"
            )

    redis_client = await get_redis_client()
    await redis_client.set_awaiting_applicant_id(session_id, False)
    await bot.send_message(chat_id, "Отслеживание прекращено")


@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    """Обработчик /reset: очищаем историю переписки для сессии."""
    if not message.from_user:
        return
    chat_id = str(message.chat.id)
    session_id = get_session_id(message)

    try:
        # Очищаем историю в Redis
        redis_client = await get_redis_client()
        await redis_client.clear_history(session_id)

        await bot.send_message(chat_id, "История переписки очищена")
    except Exception as e:
        logger.error(f"Ошибка при очистке истории для сессии {session_id}: {e}")
        await bot.send_message(chat_id, "Произошла ошибка при очистке истории")


@dp.message()
async def handle_message(message: Message):
    """Обычные сообщения: сохраняем идентификатор абитуриента или шлём в LLM."""
    if not message.from_user:
        return

    user_text = (message.text or "").strip()

    chat_id = str(message.chat.id)
    session_id = get_session_id(message)
    user_name = message.from_user.username or message.from_user.first_name or "Аноним"
    formatted_message = f"[from {user_name}] {user_text}"

    logger.info(f"Сообщение от {user_name} в чате {chat_id}: {user_text}")

    redis_client = await get_redis_client()
    is_awaiting = await redis_client.is_awaiting_applicant_id(session_id)

    # Если ожидали идентификатор абитуриента — сохраняем его
    if is_awaiting:
        async with AsyncSessionLocal() as session:
            user_service = UserService(session)
            user_id = message.from_user.id

            # Убедимся, что пользователь существует в БД
            # (на случай если /start не вызывался)
            existing_user = await user_service.get_user(user_id)
            if not existing_user:
                await user_service.create_user(user_id)
                logger.info(f"Пользователь {user_id} создан автоматически при /track")

            # Валидация формата applicant_id (макс 7 символов)
            if len(user_text) > 7:
                await bot.send_message(
                    chat_id,
                    (
                        "Идентификатор должен быть не длиннее 7 символов. "
                        "Попробуйте еще раз."
                    ),
                )
                return

            if not user_text.strip():
                await bot.send_message(
                    chat_id,
                    "Идентификатор не может быть пустым. Попробуйте еще раз.",
                )
                return

            updated = await user_service.update_applicant_id(user_id, user_text)
            if updated:
                logger.info(
                    f"Идентификатор пользователя {user_id} обновлен в БД: {user_text}"
                )
                await redis_client.set_awaiting_applicant_id(session_id, False)
                await bot.send_message(chat_id, "Идентификатор записан")
            else:
                logger.error(
                    f"Не удалось обновить идентификатор для пользователя {user_id}"
                )
                await bot.send_message(
                    chat_id,
                    (
                        "Ошибка при сохранении идентификатора. "
                        "Проверьте формат и попробуйте снова."
                    ),
                )
        return

    # Пересылаем сообщение локальной модели LLM
    await bot.send_chat_action(chat_id, "typing")
    response = await ask_local_llm(
        formatted_message, session_id=session_id, user_id=message.from_user.id
    )

    if response:
        try:
            await bot.send_message(chat_id, response, parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            plain = re.sub(r"<[^>]+>", "", response)
            await bot.send_message(chat_id, plain)


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
