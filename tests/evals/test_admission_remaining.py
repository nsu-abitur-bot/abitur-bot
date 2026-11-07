import hashlib
import json
import os
import subprocess
import sys

import httpx
import pytest

from tools.admission_index_one import SafetyCheckFailed
from tools.admission_index_remaining import (
    read_sources,
    ready_sources,
    restore_after_disconnect,
)
from tools.admission_usage import IndexingLimit, UsageLedger


def test_completed_source_is_skipped_only_for_matching_hash_and_approval():
    source = {"url": "https://example.org/one", "raw_sha256": "abc"}
    row = {
        "id": "one",
        "source_url": source["url"],
        "content_hash": "abc",
        "graph_id": "abitur_kb",
        "last_indexed_at": "2026-11-07",
        "proposal_status": "approved",
        "proposal_document_id": "one",
    }
    assert ready_sources([row], [source]) == {source["url"]}
    for field, value in [
        ("content_hash", "different"),
        ("last_indexed_at", None),
        ("proposal_status", "pending"),
        ("source_url", "foreign"),
        ("proposal_document_id", "other"),
    ]:
        with pytest.raises(SafetyCheckFailed):
            ready_sources([{**row, field: value}], [source])
    with pytest.raises(SafetyCheckFailed):
        ready_sources([row, row], [source])


def test_source_digest_and_registry_are_checked_before_indexing(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "dataset").mkdir()
    sources, registry = [], {}
    for n in range(9):
        identifier = f"DOC-{n}"
        source = {
            "id": identifier,
            "url": f"https://example.org/{n}",
            "raw_sha256": "raw",
            "filename": identifier + ".txt",
            "text_sha256": hashlib.sha256(b"text").hexdigest(),
        }
        sources.append(source)
        registry[identifier] = {"url": source["url"], "snapshot_sha256": "raw"}
        (tmp_path / "inputs" / source["filename"]).write_bytes(b"text")
    (tmp_path / "dataset/nsu_admission_2026_registry.json").write_text(
        json.dumps({"sources": registry})
    )
    (tmp_path / "batch-sources.json").write_text(json.dumps({"sources": sources}))
    assert len(read_sources(tmp_path)) == 9
    (tmp_path / "inputs/DOC-3.txt").write_bytes(b"altered")
    with pytest.raises(SafetyCheckFailed):
        read_sources(tmp_path)


def test_batch_settles_completed_cost_but_keeps_inflight_reservations(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl", batch=True)
    request = httpx.Request(
        "POST",
        "https://api.deepseek.com/chat/completions",
        json={"model": "deepseek-flash", "max_tokens": 8192},
    )
    first, second = ledger.begin(request), ledger.begin(request)
    reserve = second["reserved_micro_cny"]
    ledger.finish(
        first,
        httpx.Response(
            200,
            json={
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                }
            },
        ),
        None,
    )
    assert ledger.reserved_deepseek_micro_cny == reserve + 600
    ledger.reserved_deepseek_micro_cny = 20_000_000 - reserve + 1
    with pytest.raises(IndexingLimit):
        ledger.begin(request)


def test_batch_unknown_usage_keeps_reservation_and_blocks_spending(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl", batch=True)
    request = httpx.Request(
        "POST",
        "https://api.deepseek.com/chat/completions",
        json={"model": "deepseek-flash", "max_tokens": 8192},
    )
    row = ledger.begin(request)
    ledger.finish(row, httpx.Response(200, json={}), None)
    assert ledger.reserved_deepseek_micro_cny == row["reserved_micro_cny"]
    with pytest.raises(IndexingLimit):
        ledger.begin(request)


@pytest.mark.skipif(os.name == "nt", reason="POSIX closed pipe")
def test_lost_output_does_not_interrupt_indexing():
    command = (
        "import os; from tools.admission_index_remaining import progress; "
        "r,w=os.pipe(); os.close(r); os.dup2(w,1); os.close(w); "
        "progress({'status':'indexed'}); progress({'status':'continued'})"
    )
    result = subprocess.run([sys.executable, "-c", command], capture_output=True)
    assert result.returncode == 0, result.stderr


def test_resume_carries_all_prior_spending_and_refuses_partial_source(tmp_path):
    row = {
        "attempt": 1,
        "operation": "generation",
        "http_status": 200,
        "usage_complete": True,
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }
    payload = (json.dumps(row) + "\n").encode()
    (tmp_path / "usage.jsonl").write_bytes(payload)
    recovery = {
        "status": "recovered",
        "prior_usage_sha256": hashlib.sha256(payload).hexdigest(),
        "completed_source_ids": [str(n) for n in range(6)],
        "vector_repair_usage_sha256": hashlib.sha256(b"").hexdigest(),
    }
    recovery_path = tmp_path / "recovery.json"
    recovery_path.write_text(json.dumps(recovery))
    (tmp_path / "vector-repair-01-result.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "uncertain_failure": False,
                "attempts": 0,
                "embedding_input_bytes": 2000,
            }
        )
    )
    (tmp_path / "vector-repair-01-usage.jsonl").write_bytes(b"")
    ledger = UsageLedger(tmp_path / "next.jsonl", batch=True)
    assert restore_after_disconnect(ledger, tmp_path, recovery_path) == set(
        recovery["completed_source_ids"]
    )
    assert ledger.sequence == 1 and ledger.operation_counts["generation"] == 5
    assert ledger.reserved_deepseek_micro_cny == 5000600
    assert ledger.reserved_embedding_bytes == 5002000
    for change in (
        {"status": "started"},
        {"prior_usage_sha256": "different"},
        {"completed_source_ids": []},
    ):
        recovery_path.write_text(json.dumps({**recovery, **change}))
        with pytest.raises(SafetyCheckFailed):
            restore_after_disconnect(ledger, tmp_path, recovery_path)
