import fcntl
import logging
import re
import tempfile
from os import getenv
from pathlib import Path
from urllib.parse import urlsplit

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv

from bot.core import BotCore
from bot.utils import normalize_links_for_messaging

logger = logging.getLogger(__name__)


class _TelegramPollingLock:
    """Процессный lock, чтобы не запускать два polling на одном хосте."""

    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self._fd = None

    def acquire(self) -> bool:
        Path(self.lock_path).parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(self.lock_path, "w")
        try:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore
            self._fd.write("telegram-polling-lock\n")
            self._fd.flush()
            return True
        except BlockingIOError:
            return False

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)  # type: ignore
        finally:
            self._fd.close()
            self._fd = None


def _mask_proxy_url(proxy_url: str) -> str:
    """Возвращает URL прокси без пароля, чтобы безопасно писать в логи."""
    try:
        parsed = urlsplit(proxy_url)
        host = parsed.hostname or "unknown-host"
        port = parsed.port or "unknown-port"
        username = parsed.username
        credentials = f"{username}:***@" if username else ""
        return f"{parsed.scheme}://{credentials}{host}:{port}"
    except Exception:
        return "invalid-proxy-url"


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


async def run_telegram_bot() -> bool | None:
    """Запускает Telegram-адаптер на aiogram. Возвращает False, если запуск отменен."""
    load_dotenv()

    lock_path = getenv(
        "TELEGRAM_POLLING_LOCK_FILE",
        str(Path(tempfile.gettempdir()) / "abitur_telegram_polling.lock"),
    )
    polling_lock = _TelegramPollingLock(lock_path)
    if not polling_lock.acquire():
        logger.error(
            "Telegram polling уже запущен в другом процессе. "
            "Пропускаю второй экземпляр (lock: %s)",
            lock_path,
        )
        return False

    bot_token = getenv("BOT_TOKEN")
    if bot_token is None:
        logger.error("Telegram bot skipped: BOT_TOKEN не задан")
        return False

    telegram_socks5_proxy = getenv("TELEGRAM_SOCKS5_PROXY")
    if telegram_socks5_proxy:
        try:
            telegram_session = AiohttpSession(proxy=telegram_socks5_proxy)
        except Exception as exc:
            logger.exception(
                "Не удалось инициализировать SOCKS5 прокси для Telegram: %s. "
                "Запускаю без прокси.",
                exc,
            )
            bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=None))
        else:
            bot = Bot(
                token=bot_token,
                default=DefaultBotProperties(parse_mode=None),
                session=telegram_session,
            )
    else:
        bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=None))

    if telegram_socks5_proxy:
        logger.info(
            "Telegram SOCKS5 proxy включен: %s",
            _mask_proxy_url(telegram_socks5_proxy),
        )
    else:
        logger.info("Telegram SOCKS5 proxy выключен")

    core = BotCore()
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def cmd_start(message: Message):
        if not message.from_user:
            return
        reply = await core.cmd_start(
            channel="telegram",
            external_user_id=str(message.from_user.id),
            session_id=get_session_id(message),
        )
        await bot.send_message(
            str(message.chat.id),
            normalize_links_for_messaging(reply.text),
            parse_mode=ParseMode.HTML if reply.parse_mode == "HTML" else None,
        )

    @dp.message(Command("track"))
    async def cmd_track(message: Message):
        if not message.from_user:
            return
        reply = await core.cmd_track(session_id=get_session_id(message))
        await bot.send_message(
            str(message.chat.id), normalize_links_for_messaging(reply.text)
        )

    @dp.message(Command("untrack"))
    async def cmd_untrack(message: Message):
        if not message.from_user:
            return
        reply = await core.cmd_untrack(
            channel="telegram",
            external_user_id=str(message.from_user.id),
            session_id=get_session_id(message),
        )
        await bot.send_message(
            str(message.chat.id), normalize_links_for_messaging(reply.text)
        )

    @dp.message(Command("reset"))
    async def cmd_reset(message: Message):
        if not message.from_user:
            return
        try:
            reply = await core.cmd_reset(session_id=get_session_id(message))
            await bot.send_message(
                str(message.chat.id), normalize_links_for_messaging(reply.text)
            )
        except Exception as e:
            logger.error(
                "Ошибка при очистке истории %s: %s",
                get_session_id(message),
                e,
            )
            await bot.send_message(
                str(message.chat.id), "Произошла ошибка при очистке истории"
            )

    @dp.message(Command("feedback"))
    async def cmd_feedback(message: Message):
        if not message.from_user:
            return
        reply = await core.cmd_feedback(session_id=get_session_id(message))
        await bot.send_message(
            str(message.chat.id), normalize_links_for_messaging(reply.text)
        )

    @dp.message(Command("cancel"))
    async def cmd_cancel(message: Message):
        if not message.from_user:
            return
        reply = await core.cmd_cancel(session_id=get_session_id(message))
        await bot.send_message(
            str(message.chat.id), normalize_links_for_messaging(reply.text)
        )

    @dp.message()
    async def handle_message(message: Message):
        if not message.from_user:
            return

        user_text = (message.text or "").strip()
        user_name = (
            message.from_user.username or message.from_user.first_name or "Аноним"
        )
        chat_id = str(message.chat.id)
        session_id = get_session_id(message)

        await bot.send_chat_action(chat_id, "typing")
        reply = await core.handle_message(
            channel="telegram",
            external_user_id=str(message.from_user.id),
            session_id=session_id,
            user_name=user_name,
            user_text=user_text,
        )

        try:
            await bot.send_message(
                chat_id,
                normalize_links_for_messaging(reply.text),
                parse_mode=ParseMode.HTML if reply.parse_mode == "HTML" else None,
            )
        except TelegramBadRequest:
            if reply.fallback_plain_on_format_error:
                logger.warning(
                    "[%s] HTML parsing failed, sending plain text", session_id
                )
                plain = re.sub(r"<[^>]+>", "", reply.text)
                await bot.send_message(chat_id, normalize_links_for_messaging(plain))
            else:
                raise

    logger.info("Telegram bot started")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        polling_lock.release()
