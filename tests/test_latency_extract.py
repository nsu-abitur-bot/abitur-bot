from tools.latency_extract import extract


def test_extract_ignores_dialogs_and_strips_unapproved_fields():
    assert extract("INFO user=123 text=private") is None
    assert extract('latency {"request_id":"123", "process_first":true}') is None
    row = extract(
        "2026-11-01 11:00:00 - abitur.latency - INFO - latency "
        '{"request_id":null,"stage":"faq","duration_ms":1.5,'
        '"status":"ok","error_type":null,"text":"private"}'
    )
    assert row == {
        "request_id": None,
        "stage": "faq",
        "duration_ms": 1.5,
        "status": "ok",
        "error_type": None,
    }


def test_extract_rejects_malformed_or_non_numeric_payloads():
    for line in (
        "latency {}",
        "latency {broken}",
        'latency {"stage":"private text","status":"ok","duration_ms":1}',
        'latency {"stage":"faq","status":"ok","duration_ms":"secret"}',
    ):
        assert extract(line) is None


def test_registered_schema_rejects_private_text_and_unhashable_status():
    import json

    prefix = "2026-11-01 11:00:00 - abitur.latency - INFO - latency "
    base = {
        "request_id": None,
        "stage": "faq",
        "duration_ms": 1,
        "status": "ok",
        "error_type": None,
    }
    for change in (
        {"status": {}},
        {"stage": "synthetic_private_dialog"},
        {"process_first": "synthetic_private_dialog"},
        {"duration_ms": float("nan")},
        {"duration_ms": 10**400},
        {"error_type": {}},
    ):
        assert extract(prefix + json.dumps({**base, **change})) is None
    row = extract(prefix + json.dumps({**base, "error_type": "synthetic_private_dialog"}))
    assert row is not None
    assert row["error_type"] == "OtherError"
