import json

import pytest

from tools.admission_index_light import (
    GRAPH_FILES,
    REMAINING,
    approval_ready,
    backup_graph,
    carry_previous,
    proposal_layout_valid,
)
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


def test_backup_allows_the_application_to_have_cleared_its_optional_cache(tmp_path):
    root = tmp_path / "graph"
    destination = tmp_path / "backup"
    root.mkdir()
    destination.mkdir()
    expected = set(GRAPH_FILES) - {"kv_store_llm_response_cache.json"}
    for name in expected:
        (root / name).write_text("{}")
    assert set(backup_graph(root, destination)) == expected
    (root / "kv_store_full_docs.json").unlink()
    with pytest.raises(FileNotFoundError):
        backup_graph(root, destination)


def test_only_the_current_pending_proposal_may_lack_a_document():
    orphan = {"source_url": "https://example.org/current", "status": "pending"}
    assert proposal_layout_valid(7, 8, [orphan], orphan["source_url"])
    assert proposal_layout_valid(7, 7, [], orphan["source_url"])
    assert not proposal_layout_valid(7, 8, [orphan], "https://example.org/other")
    assert not proposal_layout_valid(
        7, 8, [{**orphan, "status": "rejected"}], orphan["source_url"]
    )
    assert not proposal_layout_valid(7, 9, [orphan, orphan], orphan["source_url"])
