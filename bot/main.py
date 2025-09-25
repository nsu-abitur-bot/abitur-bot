from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties
import asyncio
from os import getenv
import logging
from dotenv import load_dotenv
from llm.llm_client import ask_local_llm

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
    response = await ask_local_llm(formatted_message, session_id=session_id)

    if response:
        await bot.send_message(chat_id, response)


async def main():
    logger.info("Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен")
