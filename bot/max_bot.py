import asyncio
import logging
from os import getenv

from dotenv import load_dotenv
from maxapi import Bot, Dispatcher
from maxapi.enums.parse_mode import ParseMode
from maxapi.enums.sender_action import SenderAction
from maxapi.types.updates.message_created import MessageCreated

from bot.core import BotCore
from bot.streaming import MaxStreamer
from bot.utils import normalize_links_for_messaging

logger = logging.getLogger(__name__)


async def run_max_bot() -> bool | None:
    """Запускает MAX-адаптер в режиме long polling.

    Текущий вариант реализован как безопасный scaffold: если библиотека или токен
    не настроены, процесс не падает и Telegram продолжает работать. Возвращает False
    для отмены перезапуска.
    """
    load_dotenv()

    token = getenv("MAX_BOT_TOKEN")
    if not token:
        logger.info("MAX bot skipped: MAX_BOT_TOKEN is not configured")
        return False

    client = Bot(token=token, parse_mode=ParseMode.HTML)
    dp = Dispatcher()
    core = BotCore()

    logger.info("MAX bot started (long polling)")

    @dp.message_created()
    async def on_message(event: MessageCreated):
        message = event.message
        sender = message.sender
        body = message.body
        if sender is None or body is None or not body.text:
            return

        user_text = body.text.strip()
        if not user_text:
            return

        user_name = sender.username or sender.first_name or "Пользователь"
        external_user_id = str(sender.user_id)
        chat_id = message.recipient.chat_id
        session_id = str(chat_id if chat_id is not None else sender.user_id)

        if user_text == "/start":
            reply = await core.cmd_start("max", external_user_id, session_id)
            await message.answer(text=reply.text)
            return

        if user_text == "/track":
            reply = await core.cmd_track(session_id)
            await message.answer(text=reply.text)
            return

        if user_text == "/untrack":
            reply = await core.cmd_untrack("max", external_user_id, session_id)
            await message.answer(text=reply.text)
            return

        if user_text == "/reset":
            try:
                reply = await core.cmd_reset(session_id)
                await message.answer(text=reply.text)
            except Exception as exc:
                logger.error("MAX reset failed for session %s: %s", session_id, exc)
                await message.answer(text="Произошла ошибка при очистке истории")
            return

        if user_text == "/feedback":
            reply = await core.cmd_feedback(session_id)
            await message.answer(text=reply.text)
            return

        if user_text == "/cancel":
            reply = await core.cmd_cancel(session_id)
            await message.answer(text=reply.text)
            return

        # show typing action similar to Telegram
        try:
            # chat_id may be None for some update types, guard against that
            if chat_id is not None:
                await client.send_action(chat_id=chat_id, action=SenderAction.TYPING_ON)
        except Exception:
            # non-fatal: continue even if action can't be sent
            logger.debug("Failed to send typing action for session %s", session_id)

        streamer: MaxStreamer | None = (
            MaxStreamer(client, chat_id) if chat_id is not None else None
        )
        reply = await core.handle_message(
            channel="max",
            external_user_id=external_user_id,
            session_id=session_id,
            user_name=user_name,
            user_text=user_text,
            stream_callback=streamer.update if streamer else None,
            status_callback=streamer.set_status if streamer else None,
        )
        if streamer is not None:
            await streamer.finalize(
                text=reply.text,
                parse_mode=reply.parse_mode,
                fallback_plain=reply.fallback_plain_on_format_error,
            )
        else:
            await message.answer(text=normalize_links_for_messaging(reply.text))

        # Уступка API лимитам до внедрения адаптивного backoff.
        await asyncio.sleep(0)

    try:
        await dp.start_polling(client)
    finally:
        await client.close_session()
