import asyncio
import logging
from os import getenv

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Message
from dotenv import load_dotenv

from llm.llm_client import ask_local_llm, ask_local_llm_stream

load_dotenv()
BOT_TOKEN = getenv("BOT_TOKEN")
if BOT_TOKEN is None:
    raise ValueError("BOT_TOKEN не задан")

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))
dp = Dispatcher()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dp.message()
async def handle_message(message: Message):
    if not message.from_user:
        return

    user_text = message.text or ""
    chat_id = str(message.chat.id)
    user_id = str(message.from_user.id)
    session_id = (
        f"{chat_id}:{user_id}"
        if message.chat.type in ["group", "supergroup"]
        else chat_id
    )
    user_name = message.from_user.username or message.from_user.first_name or "Аноним"
    formatted_message = f"[from {user_name}] {user_text}"

    logger.info(f"Сообщение от {user_name} в чате {chat_id}: {user_text}")

    await bot.send_chat_action(chat_id, "typing")
    
    # Отправляем начальное сообщение
    sent_message = await bot.send_message(chat_id, "▌")
    
    full_response = ""
    last_update_length = 0
    update_threshold = 15  # Обновляем сообщение каждые N символов
    
    try:
        async for chunk in ask_local_llm_stream(formatted_message, session_id=session_id):
            full_response += chunk
            
            # Обновляем сообщение только если накопилось достаточно символов
            if len(full_response) - last_update_length >= update_threshold:
                try:
                    await bot.edit_message_text(
                        text=full_response + "▌",
                        chat_id=chat_id,
                        message_id=sent_message.message_id,
                    )
                    last_update_length = len(full_response)
                except Exception as e:
                    # Игнорируем ошибки редактирования (например, если текст не изменился)
                    logger.debug(f"Ошибка редактирования сообщения: {e}")
        
        # Финальное обновление без курсора
        if full_response:
            await bot.edit_message_text(
                text=full_response,
                chat_id=chat_id,
                message_id=sent_message.message_id,
            )
        else:
            await bot.edit_message_text(
                text="Ответ не найден",
                chat_id=chat_id,
                message_id=sent_message.message_id,
            )
    except Exception as e:
        logger.error(f"Ошибка стриминга: {e}")
        await bot.edit_message_text(
            text="Что-то пошло не так",
            chat_id=chat_id,
            message_id=sent_message.message_id,
        )


async def main():
    logger.info("Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен")
