import pytest

from tools.release_metrics import availability_summary, request_summary, usage_summary


def test_failures_and_timeouts_remain_in_latency_and_error_denominator():
    result = request_summary(
        [
            {"status": "ok", "duration_ms": 1},
            {"status": "error", "duration_ms": 2},
            {"status": "timeout", "duration_ms": 31000},
        ]
    )
    assert result["total"] == 3
    assert result["error_rate"] == 2 / 3
    assert result["p95_ms"] == 31000
    assert result["over_30_seconds"] == 1


def test_monitoring_gap_is_not_uptime():
    result = availability_summary(
        period_seconds=100, available_seconds=90, unavailable_seconds=0
    )
    assert result["availability_percent"] is None
    assert result["unobserved_seconds"] == 10


def test_retry_without_usage_does_not_become_zero_cost():
    attempt = dict(
        attempt_id="a",
        purpose="question",
        provider="fixture",
        model="fixture",
        kind="generation",
        input_tokens=None,
        output_tokens=None,
    )
    summary = usage_summary([attempt])
    assert summary[0]["input_tokens"] is None
    assert summary[0]["cost"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        usage_summary([attempt, attempt])


def test_question_indexing_and_judge_costs_stay_separate():
    rows = [
        dict(
            attempt_id=purpose,
            purpose=purpose,
            provider="fixture",
            model="fixture",
            kind="generation",
            input_tokens=1000,
            output_tokens=100,
            input_rate_per_million="2",
            output_rate_per_million="4",
            price_source="synthetic tariff",
            price_checked_at="2026-11-03",
            currency="TEST",
        )
        for purpose in ("question", "indexing", "judge")
    ]
    summary = usage_summary(rows)
    assert len(summary) == 3
    assert all(row["cost"] == "0.0024" for row in summary)


def test_invalid_observations_are_rejected():
    with pytest.raises(ValueError):
        request_summary([{"status": "ok", "duration_ms": float("nan")}])
    with pytest.raises(ValueError):
        availability_summary(
            period_seconds=10, available_seconds=9, unavailable_seconds=2
        )
