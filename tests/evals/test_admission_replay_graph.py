import json

import pytest

from tools.admission_index_one import SafetyCheckFailed
from tools.admission_replay_graph import carry_usage
from tools.admission_usage import UsageLedger


def test_graph_replay_preserves_prior_cost_and_rejects_unknown_usage(tmp_path):
    (tmp_path / "remaining-eight-01").mkdir()
    first = {
        "operation": "generation",
        "http_status": 200,
        "usage_complete": True,
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }
    second = {"operation": "embedding", "http_status": 200, "usage_complete": True}
    (tmp_path / "remaining-eight-01/usage.jsonl").write_text(json.dumps(first) + "\n")
    repair_usage = tmp_path / "vector-repair-01-usage.jsonl"
    repair_usage.write_text(json.dumps(second) + "\n")
    (tmp_path / "vector-repair-01-result.json").write_text(
        json.dumps(
            {
                "check": "Missing graph nodes",
                "uncertain_failure": False,
                "embedding_input_bytes": 2000,
            }
        )
    )
    ledger = UsageLedger(tmp_path / "new.jsonl", batch=True)
    assert carry_usage(ledger, tmp_path) == 2
    assert ledger.sequence == 2
    assert ledger.reserved_deepseek_micro_cny == 5000600
    assert ledger.reserved_embedding_bytes == 5002000
    assert ledger.operation_counts["generation"] == 5
    repair_usage.write_text(json.dumps({**second, "usage_complete": False}) + "\n")
    with pytest.raises(SafetyCheckFailed):
        carry_usage(ledger, tmp_path)
