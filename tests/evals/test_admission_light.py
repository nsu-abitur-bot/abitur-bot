import json

import pytest

from tools.admission_index_light import REMAINING, approval_ready, carry_previous
from tools.admission_index_one import SafetyCheckFailed
from tools.admission_usage import UsageLedger


def test_admin_approval_requires_the_exact_completed_source():
    source = {"url": "https://example.org/source", "raw_sha256": "raw"}
    row = {
        "last_indexed_at": "today",
        "content_hash": "raw",
        "source_url": source["url"],
        "source_proposal_id": "proposal",
        "rag_doc_id": "graph-document",
        "graph_id": "abitur_kb",
    }
    assert approval_ready(row, source, "proposal", "graph-document")
    for field, value in (
        ("last_indexed_at", None),
        ("content_hash", "wrong"),
        ("source_url", "wrong"),
        ("source_proposal_id", "other"),
        ("rag_doc_id", "other"),
        ("graph_id", "other"),
    ):
        assert not approval_ready(
            {**row, field: value}, source, "proposal", "graph-document"
        )


def test_light_profile_carries_both_interruption_reserves_and_all_recorded_cost(tmp_path):
    for directory in ("remaining-eight-01", "remaining-eight-02"):
        (tmp_path / directory).mkdir()
    row = {
        "operation": "generation",
        "http_status": 200,
        "usage_complete": True,
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }
    for name in (
        "remaining-eight-01/usage.jsonl",
        "vector-repair-01-usage.jsonl",
        "graph-replay-01-usage.jsonl",
        "remaining-eight-02/usage.jsonl",
    ):
        (tmp_path / name).write_text(json.dumps(row) + "\n")
    ledger = UsageLedger(
        tmp_path / "new.jsonl", batch=True, embedding_byte_limit=20000000
    )
    assert carry_previous(ledger, REMAINING[0], tmp_path) == 4
    assert ledger.sequence == 4
    assert ledger.operation_counts["generation"] == 12
    assert ledger.reserved_deepseek_micro_cny == 10002400
    assert ledger.reserved_embedding_bytes == 10000000
    (tmp_path / "remaining-eight-02/usage.jsonl").write_text(
        json.dumps({**row, "usage_complete": False}) + "\n"
    )
    with pytest.raises(SafetyCheckFailed):
        carry_previous(ledger, REMAINING[0], tmp_path)
