"""Алерты дежурному: проверки и кулдаун повторов (#243)."""

import json

import pytest

from scheduler import monitor


def _stats_body(questions: int, errors: int) -> str:
    return json.dumps(
        {
            "total_questions": questions,
            "total_answers": questions,
            "total_no_answer": 0,
            "no_answer_share": 0.0,
            "total_errors": errors,
            "error_share": errors / questions if questions else None,
            "buckets": [],
        }
    )


def _fetch(health_status=200, stats_status=200, stats_body=None, fail_urls=()):
    """Фейковый fetch: ответы по URL, часть URL может падать как сеть."""

    async def fetch(url: str) -> tuple[int, str]:
        if any(url.startswith(prefix) for prefix in fail_urls):
            raise ConnectionError("refused")
        if url.endswith("/health"):
            return health_status, ""
        return stats_status, stats_body if stats_body is not None else _stats_body(0, 0)

    return fetch


async def test_healthy_service_reports_nothing():
    assert await monitor.check_once(_fetch(), "http://api:8000") == []


async def test_api_down_reports_health_problem():
    problems = await monitor.check_once(
        _fetch(fail_urls=("http://api:8000/health",)), "http://api:8000"
    )

    assert len(problems) == 1
    assert "API не отвечает" in problems[0]


async def test_api_wrong_status_reports_health_problem():
    problems = await monitor.check_once(_fetch(health_status=503), "http://api:8000")

    assert len(problems) == 1
    assert "503" in problems[0]


async def test_error_growth_above_threshold_reports_share():
    problems = await monitor.check_once(
        _fetch(stats_body=_stats_body(questions=10, errors=3)), "http://api:8000"
    )

    assert len(problems) == 1
    assert "3 из 10" in problems[0]
    assert "30%" in problems[0]


async def test_error_growth_below_min_questions_is_silent():
    # 1 ошибка из 2 вопросов — шум, не рост ошибок.
    problems = await monitor.check_once(
        _fetch(stats_body=_stats_body(questions=2, errors=2)), "http://api:8000"
    )

    assert problems == []


async def test_error_growth_below_threshold_is_silent():
    problems = await monitor.check_once(
        _fetch(stats_body=_stats_body(questions=100, errors=5)), "http://api:8000"
    )

    assert problems == []


async def test_stats_endpoint_failure_reports_problem():
    problems = await monitor.check_once(
        _fetch(fail_urls=("http://api:8000/api/v1/logs/answer-stats",)),
        "http://api:8000",
    )

    assert len(problems) == 1
    assert "Метрики ответов недоступны" in problems[0]


async def test_independent_problems_reported_together():
    problems = await monitor.check_once(
        _fetch(
            health_status=500,
            stats_body=_stats_body(questions=10, errors=9),
        ),
        "http://api:8000",
    )

    assert len(problems) == 2


def test_alert_state_cooldown():
    state = monitor.AlertState(cooldown_seconds=100.0)

    assert state.should_send("api down", now=0.0) is True
    # Повтор внутри кулдауна подавлен, другая проблема — нет.
    assert state.should_send("api down", now=50.0) is False
    assert state.should_send("ошибки растут", now=60.0) is True
    # Кулдаун истёк — повтор уходит.
    assert state.should_send("api down", now=150.0) is True


async def test_send_alert_skipped_without_chat(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setenv("ALERT_CHAT_ID", "")
    monkeypatch.delenv("ALERT_BOT_TOKEN", raising=False)
    monkeypatch.setenv("BOT_TOKEN", "")

    with caplog.at_level("WARNING"):
        await monitor.send_alert("тест")

    assert "некуда" in caplog.text
