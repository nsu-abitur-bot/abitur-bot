from tools.latency_summarize import summarize


def test_production_attempt_count_is_unknown_without_sdk_instrumentation():
    rows = [
        {
            "request_id": "a" * 32,
            "stage": "telegram_total",
            "duration_ms": 4000,
            "status": "ok",
        },
        {
            "request_id": "a" * 32,
            "stage": "generation_tools_total",
            "duration_ms": 3000,
            "status": "ok",
        },
    ]
    request = summarize(rows)["requests"][0]
    assert request["http_attempts"] is None
    assert request["sdk_retries"] is None


def test_explicit_instrumentation_can_report_zero_attempts():
    rows = [
        {
            "request_id": "a" * 32,
            "stage": "telegram_total",
            "duration_ms": 40,
            "status": "ok",
        },
        {"request_id": "a" * 32, "event": "sdk_call", "attempts": 0, "retries": 0},
    ]
    assert summarize(rows)["requests"][0]["http_attempts"] == 0
