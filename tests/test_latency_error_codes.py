from tools.latency_error_codes import extract


def test_error_codes_strip_private_bodies_and_session_ids():
    row = extract(
        "2026-11-01 12:03:05 - pipeline.llm_client - WARNING - "
        "[synthetic-private-id] Provider generation error: "
        '429 RESOURCE_EXHAUSTED. {"message":"synthetic-private-body"}'
    )
    assert row == {
        "logged_at_utc": "2026-11-01 12:03:05",
        "component": "generation",
        "reported_http_code": 429,
    }
    assert "private" not in str(row)


def test_only_recognized_logger_and_error_prefixes_are_read():
    for line in (
        "2026-11-01 12:03:05 - bot.core - INFO - 429 RESOURCE_EXHAUSTED. private",
        "2026-11-01 12:03:05 - pipeline.llm_client - WARNING - private 429",
        "2026-99-01 12:03:05 - rag.crag - WARNING - "
        "CRAG: ошибка грейдинга чанка #1: 429 RESOURCE_EXHAUSTED. private",
    ):
        assert extract(line) is None


def test_google_error_without_status_enum_still_exposes_only_numeric_code():
    row = extract(
        "2026-11-01 12:03:05 - pipeline.llm_client - WARNING - "
        "[synthetic-private-id] Provider generation error: "
        '429 None. {"private":"body"}'
    )
    assert row == {
        "logged_at_utc": "2026-11-01 12:03:05",
        "component": "generation",
        "reported_http_code": 429,
    }
