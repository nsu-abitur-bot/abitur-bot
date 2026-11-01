import asyncio
from typing import Any, cast

from bot.streaming import TelegramStreamer


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


async def test_draft_failure_does_not_prevent_final_send():
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
