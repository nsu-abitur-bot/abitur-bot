import asyncio
import json
import logging

import pytest

from config.latency import request_timed, stage, timed


def records(caplog):
    return [
        json.loads(record.getMessage().split(" ", 1)[1])
        for record in caplog.records
        if record.name == "abitur.latency"
    ]


async def test_nested_request_uses_outer_id(caplog):
    @request_timed("pipeline")
    async def pipeline():
        return "answer"

    @request_timed("telegram")
    async def telegram():
        return await pipeline()

    with caplog.at_level(logging.INFO, logger="abitur.latency"):
        assert await telegram() == "answer"
    rows = records(caplog)
    assert len({row["request_id"] for row in rows}) == 1
    assert [row.get("stage") for row in rows] == [None, "pipeline", "telegram"]


async def test_timings_keep_request_ids_isolated_and_cross_threads(caplog):
    @request_timed("request")
    async def request(secret):
        @timed("child")
        async def child():
            await asyncio.sleep(0)

        def worker():
            with stage("thread"):
                pass

        await child()
        await asyncio.to_thread(worker)
        return secret

    with caplog.at_level(logging.INFO, logger="abitur.latency"):
        assert await asyncio.gather(request("secret-a"), request("secret-b")) == [
            "secret-a",
            "secret-b",
        ]
    rows = records(caplog)
    ids = {row["request_id"] for row in rows}
    assert len(ids) == 2 and None not in ids
    for request_id in ids:
        assert {row.get("stage") for row in rows if row["request_id"] == request_id} == {
            None,
            "request",
            "child",
            "thread",
        }
    assert "secret-" not in caplog.text


@pytest.mark.parametrize("error", [TimeoutError("private-key"), asyncio.CancelledError()])
async def test_failed_and_cancelled_operations_are_visible_without_exception_text(
    error, caplog
):
    @request_timed("request")
    async def fail():
        with stage("provider"):
            raise error

    with caplog.at_level(logging.INFO, logger="abitur.latency"):
        with pytest.raises(type(error)):
            await fail()
        with stage("outside"):
            pass
    rows = records(caplog)
    assert rows[-1]["request_id"] is None
    provider = next(row for row in rows if row.get("stage") == "provider")
    assert provider["error_type"] == type(error).__name__
    assert provider["status"] == (
        "cancelled" if isinstance(error, asyncio.CancelledError) else "error"
    )
    assert provider["duration_ms"] >= 0
    assert "private-key" not in caplog.text
