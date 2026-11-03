import asyncio
import json
import logging
from typing import Any, cast

from bot.streaming import MaxStreamer, TelegramStreamer


async def test_slow_drafts_do_not_block_generation_or_final_delivery():
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    sent = []

    class SlowBot:
        async def send_message_draft(self, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def send_message(self, chat_id, text, **kwargs):
            sent.append(text)

    streamer = TelegramStreamer(cast(Any, SlowBot()), "100", min_interval=0)
    # A stalled proxy must not stall the model callback or final send.
    async with asyncio.timeout(1):
        await streamer.set_status("Поиск…")
        await entered.wait()
        await streamer.update("Первый фрагмент")
        await streamer.update("Полный ответ")
        await streamer.finalize("Полный ответ")

    assert sent == ["Полный ответ"]
    assert cancelled.is_set()


async def test_drafts_coalesce_and_close_discards_pending_text():
    entered = asyncio.Event()
    release = asyncio.Event()
    drafts = []

    class SlowBot:
        async def send_message_draft(self, **kwargs):
            drafts.append(kwargs["text"])
            entered.set()
            await release.wait()

    streamer = TelegramStreamer(cast(Any, SlowBot()), "100", min_interval=0)
    await streamer.update("one")
    await entered.wait()
    await streamer.update("two")
    await streamer.update("three")
    release.set()
    for _ in range(5):
        await asyncio.sleep(0)
    await streamer.close()
    assert drafts == ["one", "three"]
    await streamer.close()


async def test_draft_failure_does_not_prevent_final_send(caplog):
    caplog.set_level(logging.INFO, logger="abitur.latency")
    sent = []

    class FailingBot:
        async def send_message_draft(self, **kwargs):
            raise ConnectionError("proxy unavailable")

        async def send_message(self, chat_id, text, **kwargs):
            sent.append(text)

    streamer = TelegramStreamer(cast(Any, FailingBot()), "100")
    await streamer.set_status("Поиск…")
    await asyncio.sleep(0)
    await streamer.finalize("Ответ")
    assert sent == ["Ответ"]
    rows = [
        json.loads(record.getMessage().split(" ", 1)[1])
        for record in caplog.records
        if record.name == "abitur.latency"
    ]
    draft = next(row for row in rows if row.get("stage") == "telegram_draft")
    assert draft["status"] == "error"
    assert draft["error_type"] == "ConnectionError"
    assert "proxy unavailable" not in caplog.text


async def test_finalize_preserves_cancellation_during_draft_cleanup():
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    sent = []

    class CleaningBot:
        async def send_message_draft(self, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await asyncio.Event().wait()

        async def send_message(self, chat_id, text, **kwargs):
            sent.append(text)

    streamer = TelegramStreamer(cast(Any, CleaningBot()), "100")
    await streamer.set_status("Поиск…")
    await entered.wait()
    final = asyncio.create_task(streamer.finalize("Ответ"))
    await cleaning.wait()
    final.cancel()
    try:
        await final
    except asyncio.CancelledError:
        pass
    assert final.cancelled()
    assert sent == []
    await streamer.close()


async def test_telegram_rating_buttons_are_under_final_answer_chunk():
    sent = []

    class CaptureBot:
        async def send_message(self, chat_id, text, **kwargs):
            sent.append((text, kwargs.get("reply_markup")))

    await TelegramStreamer(cast(Any, CaptureBot()), "100").finalize(
        "Ответ. " * 1000, feedback_id="candidate-1"
    )

    assert len(sent) > 1
    assert all(markup is None for _, markup in sent[:-1])
    buttons = sent[-1][1].inline_keyboard[0]
    assert [button.callback_data for button in buttons] == [
        "rate:positive:candidate-1",
        "rate:negative:candidate-1",
    ]


async def test_max_rating_buttons_are_under_final_answer_chunk():
    sent = []

    class CaptureClient:
        async def send_message(self, **kwargs):
            sent.append(kwargs)
            return None

    await MaxStreamer(cast(Any, CaptureClient()), 100).finalize(
        "Ответ. " * 1000, feedback_id="candidate-2"
    )

    assert len(sent) > 1
    assert all(message["attachments"] is None for message in sent[:-1])
    buttons = sent[-1]["attachments"][0].payload.buttons[0]
    assert [button.payload for button in buttons] == [
        "rate:positive:candidate-2",
        "rate:negative:candidate-2",
    ]
