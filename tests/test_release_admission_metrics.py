import copy
import json
from decimal import Decimal

import pytest

from tools.release_admission_metrics import calculate, summarize_usage


def test_real_evidence_phases_reconcile_without_counting_judge_as_answers():
    result = calculate()
    assert len(result["verified_inputs"]) == 73
    total = result["total_usage"]
    assert total["http_attempts"] == 1826
    assert total["deepseek_generations"] == 1375
    assert total["deepseek_historical_cny"] == "0.74856644"
    assert total["gigachat_embedding_tokens"] == 18131
    assert total["gigachat_cost"] is None
    phases = [
        phase
        for condition in result["conditions"].values()
        for phase in condition["usage_by_phase"].values()
    ]
    assert sum(p["http_attempts"] for p in phases) == total["http_attempts"]
    assert sum(Decimal(p["deepseek_historical_cny"]) for p in phases) == Decimal(
        total["deepseek_historical_cny"]
    )
    assert (
        sum(
            c["usage_by_phase"]["judge"]["deepseek_generations"]
            for c in result["conditions"].values()
        )
        == 99
    )
    assert all(c["bot_only_latency"] is None for c in result["conditions"].values())
    assert result["quality"]["conditions"]["with_kb"]["reviewed"]["total"]["passed"] == 30


def test_missing_or_inconsistent_usage_cannot_become_zero_cost():
    row = {
        "usage_complete": True,
        "http_status": 200,
        "error_type": None,
        "operation": "generation",
        "provider": "deepseek",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_cache_hit_tokens": 4,
            "prompt_cache_miss_tokens": 6,
        },
    }
    assert summarize_usage([row])["deepseek_historical_cny"] == "0.00001408"
    for change in ("missing", "mismatch", "incomplete", "failed"):
        invalid = copy.deepcopy(row)
        if change == "missing":
            invalid["usage"].pop("prompt_cache_hit_tokens")
        elif change == "mismatch":
            invalid["usage"]["total_tokens"] = 11
        elif change == "incomplete":
            invalid["usage_complete"] = False
        else:
            invalid["http_status"] = 500
        with pytest.raises(ValueError):
            summarize_usage([invalid])


def test_modified_evidence_is_rejected_before_calculation(tmp_path, monkeypatch):
    from tools import release_admission_metrics

    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": [{"file": "sample.json", "sha256": "0" * 64}]}),
        encoding="utf-8",
    )
    (tmp_path / "sample.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(release_admission_metrics, "DATA", tmp_path)
    with pytest.raises(ValueError, match="Evidence SHA mismatch"):
        calculate()
