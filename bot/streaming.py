"""Утилита для прогрессивного редактирования одного Telegram/MAX-сообщения."""

import logging
import re
import time
from typing import TYPE_CHECKING, Optional

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest

from bot.utils import normalize_links_for_messaging

if TYPE_CHECKING:
    from maxapi import Bot as MaxBot

logger = logging.getLogger(__name__)

DEFAULT_MIN_EDIT_INTERVAL = 1.2  # Telegram рекомендует ≤ 1 edit/сек на сообщение


class TelegramStreamer:
    """Шлёт первое сообщение и затем троттлит edit_message_text.

    Промежуточные правки идут plain text (без parse_mode), потому что
    частичный HTML с незакрытыми тегами (<b>, <a>) валит Telegram-парсер.
    Финальное сообщение перерисовывается с нужным parse_mode в finalize().
    """

    def __init__(
        self,
        bot: Bot,
        chat_id: str,
        min_interval: float = DEFAULT_MIN_EDIT_INTERVAL,
    ) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._min_interval = min_interval
        self._message_id: Optional[int] = None
        self._last_sent_text: str = ""
        self._last_edit_monotonic: float = 0.0

    async def set_status(self, text: str) -> None:
        """Мгновенно ставит текст-статус (без throttle и без parse_mode).

        Используется для промежуточных фраз вроде "Поиск в базе знаний…": между
        статусами проходит сотни мс — секунды, throttle здесь только мешает.
        Обновляет _last_edit_monotonic, чтобы первый последующий update() не
        выстрелил мгновенно и не словил Telegram rate-limit.
        """
        plain = normalize_links_for_messaging((text or "").strip())
        if not plain:
            return

        if self._message_id is None:
            try:
                msg = await self._bot.send_message(self._chat_id, plain)
            except Exception as exc:
                logger.warning(
                    "TelegramStreamer: failed to send status message: %s", exc
                )
                return
            self._message_id = msg.message_id
            self._last_sent_text = plain
            self._last_edit_monotonic = time.monotonic()
            return

        if plain == self._last_sent_text:
            return

        try:
            await self._bot.edit_message_text(
                text=plain,
                chat_id=self._chat_id,
                message_id=self._message_id,
            )
            self._last_sent_text = plain
            self._last_edit_monotonic = time.monotonic()
        except TelegramBadRequest as exc:
            if "not modified" not in str(exc).lower():
                logger.debug("TelegramStreamer: status edit failed (%s)", exc)
        except Exception as exc:
            logger.debug("TelegramStreamer: status edit failed (%s)", exc)

    async def update(self, text: str) -> None:
        """Принимает накопленный текст ответа LLM. Безопасно для частых вызовов."""
        text = (text or "").strip()
        if not text:
            return

        plain = normalize_links_for_messaging(text)

        if self._message_id is None:
            try:
                msg = await self._bot.send_message(self._chat_id, plain)
            except Exception as exc:
                logger.warning(
                    "TelegramStreamer: failed to send initial message: %s", exc
                )
                return
            self._message_id = msg.message_id
            self._last_sent_text = plain
            self._last_edit_monotonic = time.monotonic()
            return

        now = time.monotonic()
        if now - self._last_edit_monotonic < self._min_interval:
            return
        if plain == self._last_sent_text:
            return

        try:
            await self._bot.edit_message_text(
                text=plain,
                chat_id=self._chat_id,
                message_id=self._message_id,
            )
            self._last_sent_text = plain
            self._last_edit_monotonic = now
        except TelegramBadRequest as exc:
            # "message is not modified" безобидно — просто пропускаем
            if "not modified" not in str(exc).lower():
                logger.debug("TelegramStreamer: edit failed (%s)", exc)
        except Exception as exc:
            logger.debug("TelegramStreamer: edit failed (%s)", exc)

    async def finalize(
        self,
        text: str,
        parse_mode: Optional[str] = None,
        fallback_plain: bool = False,
    ) -> None:
        """Принудительный финальный send/edit с нужным parse_mode."""
        final_text = normalize_links_for_messaging(text)
        aiogram_parse_mode = ParseMode.HTML if parse_mode == "HTML" else None

        if self._message_id is None:
            # стрима не было (FAQ / awaiting / пустой стрим) — шлём как обычное сообщение
            try:
                msg = await self._bot.send_message(
                    self._chat_id, final_text, parse_mode=aiogram_parse_mode
                )
                self._message_id = msg.message_id
                self._last_sent_text = final_text
                return
            except TelegramBadRequest:
                if not fallback_plain:
                    raise
                logger.warning(
                    "TelegramStreamer: HTML parse failed on send, "
                    "sending plain text"
                )
                plain = re.sub(r"<[^>]+>", "", final_text)
                msg = await self._bot.send_message(
                    self._chat_id, normalize_links_for_messaging(plain)
                )
                self._message_id = msg.message_id
                return

        try:
            await self._bot.edit_message_text(
                text=final_text,
                chat_id=self._chat_id,
                message_id=self._message_id,
                parse_mode=aiogram_parse_mode,
            )
            self._last_sent_text = final_text
        except TelegramBadRequest as exc:
            err = str(exc).lower()
            if "not modified" in err:
                return
            if not fallback_plain:
                raise
            logger.warning(
                "TelegramStreamer: HTML parse failed on edit, falling back to plain"
            )
            plain = re.sub(r"<[^>]+>", "", final_text)
            try:
                await self._bot.edit_message_text(
                    text=normalize_links_for_messaging(plain),
                    chat_id=self._chat_id,
                    message_id=self._message_id,
                )
            except TelegramBadRequest as exc2:
                if "not modified" not in str(exc2).lower():
                    raise


class MaxStreamer:
    """Аналог TelegramStreamer для MAX: одно сообщение, статус-фразы через edit.

    Token-стрим LLM в MAX не делаем (update() — no-op): показываем только
    статусы пайплайна и финальный ответ одним куском, оба идут как edit одного
    и того же сообщения.
    """

    def __init__(self, client: "MaxBot", chat_id: int) -> None:
        self._client = client
        self._chat_id = chat_id
        self._message_id: Optional[str] = None
        self._last_sent_text: str = ""

    @staticmethod
    def _resolve_parse_mode(parse_mode: Optional[str]):
        if parse_mode != "HTML":
            return None
        try:
            from maxapi.enums.parse_mode import ParseMode as MaxParseMode
        except ImportError:
            return None
        return MaxParseMode.HTML

    async def set_status(self, text: str) -> None:
        plain = normalize_links_for_messaging((text or "").strip())
        if not plain:
            return

        if self._message_id is None:
            try:
                sent = await self._client.send_message(
                    chat_id=self._chat_id, text=plain
                )
            except Exception as exc:
                logger.warning("MaxStreamer: failed to send status message: %s", exc)
                return
            if sent is None or sent.message is None:
                return
            self._message_id = sent.message.body.mid
            self._last_sent_text = plain
            return

        if plain == self._last_sent_text:
            return

        try:
            await self._client.edit_message(
                message_id=self._message_id, text=plain
            )
            self._last_sent_text = plain
        except Exception as exc:
            logger.debug("MaxStreamer: status edit failed (%s)", exc)

    async def update(self, text: str) -> None:  # noqa: ARG002 - intentional no-op
        """LLM-стрим в MAX не делаем — оставляем no-op, чтобы интерфейс совпадал."""
        return

    async def finalize(
        self,
        text: str,
        parse_mode: Optional[str] = None,
        fallback_plain: bool = False,
    ) -> None:
        final_text = normalize_links_for_messaging(text)
        max_parse_mode = self._resolve_parse_mode(parse_mode)

        if self._message_id is None:
            try:
                sent = await self._client.send_message(
                    chat_id=self._chat_id, text=final_text, parse_mode=max_parse_mode
                )
            except Exception:
                if not fallback_plain:
                    raise
                logger.warning(
                    "MaxStreamer: HTML parse failed on send, sending plain text"
                )
                plain = normalize_links_for_messaging(re.sub(r"<[^>]+>", "", final_text))
                sent = await self._client.send_message(
                    chat_id=self._chat_id, text=plain
                )
            if sent is not None and sent.message is not None:
                self._message_id = sent.message.body.mid
                self._last_sent_text = final_text
            return

        try:
            await self._client.edit_message(
                message_id=self._message_id,
                text=final_text,
                parse_mode=max_parse_mode,
            )
            self._last_sent_text = final_text
        except Exception:
            if not fallback_plain:
                raise
            logger.warning(
                "MaxStreamer: HTML parse failed on edit, falling back to plain"
            )
            plain = normalize_links_for_messaging(re.sub(r"<[^>]+>", "", final_text))
            try:
                await self._client.edit_message(
                    message_id=self._message_id, text=plain
                )
            except Exception as exc2:
                logger.debug("MaxStreamer: plain fallback edit failed (%s)", exc2)
