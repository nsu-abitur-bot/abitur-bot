import json

import httpx
import pytest

from tools.admission_pilot import (
    CASE_IDS,
    DATASET_SHA,
    IMAGE_REVISION,
    PilotLedger,
    validate_plan,
)
from tools.admission_usage import IndexingLimit, UsageLedger


def test_pilot_plan_refuses_full_dataset_or_changed_version():
    plan = {
        "mode": "ten_case_pilot",
        "case_ids": list(CASE_IDS),
        "dataset_sha256": DATASET_SHA,
        "image_revision": IMAGE_REVISION,
        "full_run_authorized": False,
    }
    validate_plan(plan)
    for key, value in (
        ("case_ids", list(CASE_IDS) * 5),
        ("full_run_authorized", True),
        ("image_revision", "other"),
    ):
        with pytest.raises(RuntimeError):
            validate_plan({**plan, key: value})


def test_inspection_never_sends_a_paid_request(tmp_path):
    ledger = PilotLedger(tmp_path, inspect_only=True)
    reached = []

    def unexpected_request(request):
        reached.append(request)
        return httpx.Response(500)

    transport = httpx.MockTransport(unexpected_request)
    with ledger.capture(), httpx.Client(transport=transport) as client:
        with pytest.raises(IndexingLimit):
            client.post(
                "https://api.deepseek.com/chat/completions",
                json={"model": "deepseek-flash", "max_tokens": 100},
            )
    assert reached == []


def test_intent_is_durable_before_send_and_no_payload_is_persisted(tmp_path):
    ledger = PilotLedger(tmp_path)
    ledger.context("judge", "nsu-pre-flow-01")

    def reply(request):
        intents = [
            json.loads(line) for line in ledger.intent_path.read_text().splitlines()
        ]
        assert len(intents) == 1 and intents[0]["phase"] == "judge"
        assert intents[0]["case_id"] == "nsu-pre-flow-01"
        assert ledger.completed == 0
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
            },
        )

    with ledger.capture(), httpx.Client(transport=httpx.MockTransport(reply)) as client:
        client.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": "Bearer synthetic-secret"},
            json={
                "model": "deepseek-flash",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "private-text"}],
            },
        )
    assert ledger.completed == ledger.sequence == 1
    saved = ledger.path.read_text() + ledger.intent_path.read_text()
    assert "synthetic-secret" not in saved and "private-text" not in saved
    with pytest.raises(FileExistsError):
        PilotLedger(tmp_path)


def test_price_guard_can_only_be_tightened(tmp_path):
    with pytest.raises(ValueError):
        UsageLedger(tmp_path / "usage", batch=True, deepseek_ceiling_micro_cny=21_000_000)
    ledger = UsageLedger(tmp_path / "usage", batch=True, deepseek_ceiling_micro_cny=1)
    with pytest.raises(IndexingLimit):
        ledger.begin(
            httpx.Request(
                "POST",
                "https://api.deepseek.com/chat/completions",
                json={"model": "deepseek-flash", "max_tokens": 100},
            )
        )
    assert ledger.sequence == 0
