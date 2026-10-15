"""Хендлеры Telegram и MAX: человек получает ответ при ошибке и длинном ответе."""

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendMessage

from bot.core import ERROR_REPLY_TEXT, BotReply
from bot.max_bot import answer_max_message
from bot.splitting import MESSAGE_CHUNK_LIMIT
from bot.telegram_bot import answer_message

LONG_HTML_ANSWER = "\n\n".join(
    f"<b>Пункт {i}.</b> " + "Документы подаются через Госуслуги. " * 20 for i in range(20)
)


class FakeTelegramBot:
    def __init__(self, fail_on_send: bool = False) -> None:
        self.sent: list[tuple[str, Any]] = []
        self.fail_on_send = fail_on_send

    async def send_chat_action(self, chat_id, action) -> None:
        return None

    async def send_message_draft(self, **kwargs) -> None:
        return None

    async def send_message(self, chat_id, text, parse_mode=None) -> None:
        if len(text) > 4096:
            raise TelegramBadRequest(
                method=SendMessage(chat_id=chat_id, text=text),
                message="Bad Request: message is too long",
            )
        if self.fail_on_send:
            raise RuntimeError("telegram is down")
        self.sent.append((text, parse_mode))


class FakeCore:
    def __init__(self, reply: BotReply | None = None, error: BaseException | None = None):
        self.reply = reply
        self.error = error

    async def handle_message(self, **kwargs) -> BotReply:
        if self.error is not None:
            raise self.error
        assert self.reply is not None
        return self.reply


def telegram_message() -> Any:
    return SimpleNamespace(
        text="Как поступить?",
        from_user=SimpleNamespace(id=7, username="abiturient", first_name="А"),
        chat=SimpleNamespace(id=100, type="private"),
    )


async def run_telegram(bot: FakeTelegramBot, core: FakeCore) -> None:
    await answer_message(cast(Any, bot), cast(Any, core), telegram_message())


async def test_telegram_exception_gives_fallback_reply_and_is_logged(caplog):
    bot = FakeTelegramBot()
    core = FakeCore(error=ConnectionError("database is unavailable"))

    with caplog.at_level(logging.ERROR, logger="bot.telegram_bot"):
        await run_telegram(bot, core)

    assert bot.sent == [(ERROR_REPLY_TEXT, None)]
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and errors[0].exc_info is not None
    assert "database is unavailable" in caplog.text


async def test_telegram_failing_fallback_does_not_raise(caplog):
    bot = FakeTelegramBot(fail_on_send=True)
    core = FakeCore(error=RuntimeError("boom"))

    with caplog.at_level(logging.ERROR, logger="bot.telegram_bot"):
        await run_telegram(bot, core)

    assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 2


async def test_telegram_cancellation_is_not_swallowed():
    bot = FakeTelegramBot()
    core = FakeCore(error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await run_telegram(bot, core)
    assert bot.sent == []


async def test_telegram_long_answer_arrives_in_several_messages():
    bot = FakeTelegramBot()
    reply = BotReply(
        text=LONG_HTML_ANSWER, parse_mode="HTML", fallback_plain_on_format_error=True
    )
    assert len(reply.text) > 4096

    await run_telegram(bot, FakeCore(reply=reply))

    assert len(bot.sent) > 1
    assert all(len(text) <= MESSAGE_CHUNK_LIMIT for text, _ in bot.sent)
    assert ERROR_REPLY_TEXT not in [text for text, _ in bot.sent]
    assert "Пункт 0." in bot.sent[0][0]
    assert "Пункт 19." in bot.sent[-1][0]


class FakeMaxClient:
    """MAX API: отправленное сообщение получает mid, лимит текста — < 4000."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.edits: list[tuple[str, str]] = []

    async def send_action(self, **kwargs) -> None:
        return None

    async def send_message(self, chat_id, text, parse_mode=None):
        if len(text) >= 4000:
            raise ValueError("text должен быть меньше 4000 символов")
        self.sent.append(text)
        mid = f"mid-{len(self.sent)}"
        return SimpleNamespace(message=SimpleNamespace(body=SimpleNamespace(mid=mid)))

    async def edit_message(self, message_id, text, parse_mode=None) -> None:
        if len(text) >= 4000:
            raise ValueError("text должен быть меньше 4000 символов")
        self.edits.append((message_id, text))


class StatusThenFailCore:
    """Успевает показать статус и падает — как сбой провайдера посреди поиска."""

    async def handle_message(self, **kwargs) -> BotReply:
        await kwargs["status_callback"]("Ищу в базе знаний…")
        raise RuntimeError("provider failed")


async def run_max(client: FakeMaxClient, core: Any, chat_id: int | None = 5) -> None:
    await answer_max_message(
        cast(Any, client),
        cast(Any, core),
        cast(Any, SimpleNamespace()),
        chat_id=chat_id,
        session_id="5",
        external_user_id="9",
        user_name="abiturient",
        user_text="Как поступить?",
    )


async def test_max_exception_replaces_status_with_fallback_reply(caplog):
    client = FakeMaxClient()

    with caplog.at_level(logging.ERROR, logger="bot.max_bot"):
        await run_max(client, StatusThenFailCore())

    assert client.sent == ["Ищу в базе знаний…"]
    assert client.edits == [("mid-1", ERROR_REPLY_TEXT)]
    assert "provider failed" in caplog.text


async def test_max_without_chat_id_answers_fallback_via_message():
    answered: list[str] = []

    async def answer(text: str) -> None:
        answered.append(text)

    await answer_max_message(
        cast(Any, FakeMaxClient()),
        cast(Any, FakeCore(error=RuntimeError("redis is down"))),
        cast(Any, SimpleNamespace(answer=answer)),
        chat_id=None,
        session_id="9",
        external_user_id="9",
        user_name="abiturient",
        user_text="Как поступить?",
    )

    assert answered == [ERROR_REPLY_TEXT]


async def test_max_cancellation_is_not_swallowed():
    client = FakeMaxClient()

    with pytest.raises(asyncio.CancelledError):
        await run_max(client, FakeCore(error=asyncio.CancelledError()))
    assert client.sent == []


async def test_max_long_answer_edits_status_then_sends_rest():
    client = FakeMaxClient()
    text = "\n\n".join("Документы подаются через Госуслуги. " * 20 for _ in range(20))

    class StatusThenLongCore:
        async def handle_message(self, **kwargs) -> BotReply:
            await kwargs["status_callback"]("Ищу в базе знаний…")
            return BotReply(text=text)

    await run_max(client, StatusThenLongCore())

    assert len(client.edits) == 1
    assert client.edits[0][0] == "mid-1"
    assert len(client.sent) > 1  # статус + остальные части
    parts = [client.edits[0][1], *client.sent[1:]]
    assert all(len(p) <= MESSAGE_CHUNK_LIMIT for p in parts)
    assert ERROR_REPLY_TEXT not in parts
