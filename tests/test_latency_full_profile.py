import json
from typing import Any, cast

import pytest

from tools.latency_full_profile import instrument_sdk


async def test_sdk_attempt_budget_persists_and_blocks_before_network(
    tmp_path, monkeypatch
):
    from google.genai._api_client import BaseApiClient

    sent = []

    async def fake_send(self, *args, **kwargs):
        sent.append(True)
        return "ok"

    def fake_sync(self, *args, **kwargs):
        sent.append(True)
        return "ok"

    # Register every patched attribute with monkeypatch, so the harness's
    # process-scoped instrumentation cannot leak into another test.
    for name in ("_async_request", "_request"):
        monkeypatch.setattr(BaseApiClient, name, getattr(BaseApiClient, name))
    monkeypatch.setattr(BaseApiClient, "_async_request_once", fake_send)
    monkeypatch.setattr(BaseApiClient, "_request_once", fake_sync)
    ledger = tmp_path / "budget.json"
    ledger.write_text(json.dumps({"attempts": 1, "limit": 2}))
    instrument_sdk(ledger, 2)
    assert await cast(Any, BaseApiClient)._async_request_once(object()) == "ok"
    with pytest.raises(RuntimeError, match="budget exhausted"):
        cast(Any, BaseApiClient)._request_once(object())
    assert sent == [True]
    assert json.loads(ledger.read_text())["attempts"] == 2
