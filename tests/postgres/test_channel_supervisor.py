import asyncio

import pytest

from bot import channel_supervisor
from config.instance import install


@pytest.mark.asyncio
async def test_token_replacement_stops_old_poller_before_starting_new(
    test_engine, monkeypatch
):
    monkeypatch.setattr(channel_supervisor, "engine", test_engine)
    events = asyncio.Queue()
    install(
        {
            "TELEGRAM_ENABLED": "true",
            "BOT_TOKEN": "first",
            "BOT_RESTART_DELAY_SECONDS": "0.01",
        },
        1,
    )

    async def runner():
        token = channel_supervisor.channel_signature("telegram")[0]
        await events.put(("start", token))
        try:
            await asyncio.Future()
        finally:
            await events.put(("stop", token))

    task = asyncio.create_task(channel_supervisor.supervise_channel("telegram", runner))
    try:
        assert await asyncio.wait_for(events.get(), 5) == ("start", "first")
        install(
            {
                "TELEGRAM_ENABLED": "true",
                "BOT_TOKEN": "second",
                "BOT_RESTART_DELAY_SECONDS": "0.01",
            },
            2,
        )
        assert await asyncio.wait_for(events.get(), 5) == ("stop", "first")
        assert await asyncio.wait_for(events.get(), 5) == ("start", "second")
        install({"TELEGRAM_ENABLED": "false"}, 3)
        assert await asyncio.wait_for(events.get(), 5) == ("stop", "second")
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_two_supervisors_do_not_poll_same_token(test_engine, monkeypatch):
    monkeypatch.setattr(channel_supervisor, "engine", test_engine)
    starts = []
    install(
        {
            "MAX_ENABLED": "true",
            "MAX_BOT_TOKEN": "one-token",
            "BOT_RESTART_DELAY_SECONDS": "0.01",
        },
        1,
    )

    async def runner():
        starts.append("started")
        await asyncio.Future()

    tasks = [
        asyncio.create_task(channel_supervisor.supervise_channel("max", runner))
        for _ in range(2)
    ]
    try:
        await asyncio.sleep(2.2)
        assert starts == ["started"]
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
