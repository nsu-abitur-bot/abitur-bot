import hashlib
import json

import pytest

from tools.admission_full_plan import (
    BUDGET,
    DATASET_SHA,
    IMAGE,
    batch_ids,
    prior_cost,
    reserve,
)

from tools.admission_pilot import empty_store_valid


def test_empty_control_accepts_only_keyword_search_cache():
    row = {
        "cache_type": "keywords",
        "return": json.dumps(
            {"high_level_keywords": ["поступление"], "low_level_keywords": ["ФИТ"]}
        ),
    }
    assert empty_store_valid("kv_store_llm_response_cache.json", {"one": row})
    assert not empty_store_valid("kv_store_full_docs.json", {"one": row})
    assert not empty_store_valid(
        "kv_store_llm_response_cache.json", {"one": {**row, "cache_type": "query"}}
    )
    assert not empty_store_valid(
        "kv_store_llm_response_cache.json",
        {"one": {**row, "return": '{"answer":"fact"}'}},
    )


def plan():
    ids = [f"case-{i}" for i in range(50)]
    return {
        "mode": "full_50_comparison",
        "full_run_authorized": True,
        "dataset_sha256": DATASET_SHA,
        "image_revision": IMAGE,
        "budget_micro_cny": BUDGET,
        "batch_size": 10,
        "case_ids": ids,
        "groups": {key: "document" if i < 45 else "special" for i, key in enumerate(ids)},
    }


def test_batches_cover_fifty_once_and_reject_changed_authorization():
    p = plan()
    assert [key for i in range(5) for key in batch_ids(p, i)] == p["case_ids"]
    for change in (
        {"full_run_authorized": False},
        {"case_ids": ["same"] * 50},
        {"batch_size": 50},
        {"dataset_sha256": "changed"},
    ):
        with pytest.raises(RuntimeError):
            batch_ids({**p, **change}, 0)
    with pytest.raises(RuntimeError):
        batch_ids(p, 5)


def usage():
    return {
        "attempt": 1,
        "provider": "deepseek",
        "http_status": 200,
        "error_type": None,
        "usage_complete": True,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


def test_global_budget_accounts_previous_condition_and_rejects_tampering(tmp_path):
    folder = tmp_path / "evidence"
    folder.mkdir()
    raw = (json.dumps(usage()) + "\n").encode()
    (folder / "with-kb-00.jsonl").write_bytes(raw)
    p = {
        "budget_evidence": [
            {
                "path": "evidence/with-kb-00.jsonl",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ],
        "remaining_micro_cny": BUDGET - 360,
    }
    assert prior_cost(p, tmp_path) == 360
    with pytest.raises(RuntimeError):
        prior_cost({**p, "remaining_micro_cny": BUDGET}, tmp_path)
    (folder / "with-kb-00.jsonl").write_bytes(b"changed")
    with pytest.raises(RuntimeError):
        prior_cost(p, tmp_path)


def test_unknown_usage_and_duplicate_attempts_halt_next_batch():
    for rows in (
        [{**usage(), "usage_complete": False}],
        [usage(), usage()],
        [{**usage(), "http_status": 500}],
    ):
        with pytest.raises(RuntimeError):
            reserve(rows)
