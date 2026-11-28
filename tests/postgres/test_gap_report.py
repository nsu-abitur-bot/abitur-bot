"""Интеграционные тесты отчёта о пробелах на реальной PostgreSQL (#485)."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

import api.services.gap_report as gap_report_service
from api.auth.dependencies import get_current_admin
from api.main import app
from db.postgres.db import get_async_session
from db.postgres.models import AnswerFeedback, MessageLog, Topic


def _freeze_now(monkeypatch: pytest.MonkeyPatch, now: datetime) -> None:
    monkeypatch.setattr(
        gap_report_service,
        "datetime",
        SimpleNamespace(
            now=lambda tz=None: now if tz is not None else now.replace(tzinfo=None)
        ),
    )


@pytest.mark.asyncio
async def test_gap_report_applies_days_period_to_all_sections(
    session: AsyncSession, monkeypatch
):
    """days=1 и days=30 ограничивают все 5 разделов отчёта единым периодом."""
    now = datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc)
    now_naive = now.replace(tzinfo=None)
    old_ts = now_naive - timedelta(days=5)
    recent_ts = now_naive - timedelta(hours=3)
    future_ts = now_naive + timedelta(hours=1)

    _freeze_now(monkeypatch, now)
    monkeypatch.setitem(app.dependency_overrides, get_current_admin, lambda: None)
    monkeypatch.setitem(app.dependency_overrides, get_async_session, lambda: session)

    topic_old = Topic(label="Общежитие", is_active=True)
    topic_recent = Topic(label="Документы", is_active=True)
    session.add_all([topic_old, topic_recent])
    await session.flush()

    logs: list[MessageLog] = []
    # Старый период (5 дней назад): частые популярные и неотвеченные вопросы,
    # тема «Общежитие», провал в quality.
    for i in range(4):
        logs.append(
            MessageLog(
                user_id=100 + i,
                session_id=f"old-pop-{i}",
                message_type="user_input",
                content="Старый популярный вопрос",
                topic_id=topic_old.id,
                created_at=old_ts,
            )
        )
    for i in range(3):
        logs.append(
            MessageLog(
                user_id=200 + i,
                session_id=f"old-na-{i}",
                message_type="no_answer",
                content="Старый неотвеченный вопрос",
                created_at=old_ts,
            )
        )
    logs.append(
        MessageLog(
            user_id=300,
            session_id="old-llm",
            message_type="llm_response",
            content="Не нашёл",
            message_metadata={"latency_ms": 100.0, "no_answer": True},
            created_at=old_ts,
        )
    )

    # Свежий период (3 часа назад): 1 вопрос, 1 неотвеченный, 1 ответ.
    logs.extend(
        [
            MessageLog(
                user_id=400,
                session_id="new-pop",
                message_type="user_input",
                content="Актуальный вопрос",
                topic_id=topic_recent.id,
                created_at=recent_ts,
            ),
            MessageLog(
                user_id=401,
                session_id="new-na",
                message_type="no_answer",
                content="Актуальный неотвеченный",
                created_at=recent_ts,
            ),
            MessageLog(
                user_id=400,
                session_id="new-pop",
                message_type="llm_response",
                content="Ответ",
                message_metadata={"latency_ms": 120.0, "no_answer": False},
                created_at=recent_ts,
            ),
            # Запись после верхней границы now — не должна попадать ни в один раздел.
            MessageLog(
                user_id=500,
                session_id="future",
                message_type="user_input",
                content="Вопрос из будущего",
                topic_id=topic_recent.id,
                created_at=future_ts,
            ),
            MessageLog(
                user_id=501,
                session_id="future-na",
                message_type="no_answer",
                content="Неотвеченный из будущего",
                created_at=future_ts,
            ),
            MessageLog(
                user_id=500,
                session_id="future",
                message_type="llm_response",
                content="Ошибка",
                message_metadata={"latency_ms": 200.0, "error": True},
                created_at=future_ts,
            ),
        ]
    )
    session.add_all(logs)
    session.add_all(
        [
            AnswerFeedback(
                user_id=100,
                session_id="old-fb",
                channel="telegram",
                question="Старая плохая оценка",
                bot_response="ответ",
                rating="negative",
                created_at=old_ts,
                rated_at=old_ts,
            ),
            AnswerFeedback(
                user_id=400,
                session_id="new-fb",
                channel="telegram",
                question="Свежая плохая оценка",
                bot_response="ответ",
                rating="negative",
                created_at=recent_ts,
                rated_at=recent_ts,
            ),
            AnswerFeedback(
                user_id=500,
                session_id="future-fb",
                channel="telegram",
                question="Будущая плохая оценка",
                bot_response="ответ",
                rating="negative",
                created_at=future_ts,
                rated_at=future_ts,
            ),
        ]
    )
    await session.commit()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        day_resp = await client.get("/api/v1/reports/gaps?days=1&limit=10")
        month_resp = await client.get("/api/v1/reports/gaps?days=30&limit=10")

    assert day_resp.status_code == 200
    day = day_resp.json()
    assert day["period_days"] == 1
    assert day["topics"] == [{"topic": "Документы", "count": 1}]
    assert [q["question"] for q in day["popular"]] == ["Актуальный вопрос"]
    assert [q["question"] for q in day["unanswered"]] == ["Актуальный неотвеченный"]
    assert day["quality"]["questions"] == 1
    assert day["quality"]["answers"] == 1
    assert day["quality"]["no_answer"] == 0
    assert day["quality"]["errors"] == 0
    assert [d["question"] for d in day["disliked"]] == ["Свежая плохая оценка"]

    assert month_resp.status_code == 200
    month = month_resp.json()
    assert month["period_days"] == 30
    assert {t["topic"]: t["count"] for t in month["topics"]} == {
        "Общежитие": 4,
        "Документы": 1,
    }
    assert [q["question"] for q in month["popular"]] == [
        "Старый популярный вопрос",
        "Актуальный вопрос",
    ]
    assert [q["question"] for q in month["unanswered"]] == [
        "Старый неотвеченный вопрос",
        "Актуальный неотвеченный",
    ]
    assert month["quality"]["questions"] == 5
    assert month["quality"]["answers"] == 2
    assert month["quality"]["no_answer"] == 1
    assert month["quality"]["errors"] == 0
    assert {d["question"] for d in month["disliked"]} == {
        "Старая плохая оценка",
        "Свежая плохая оценка",
    }


@pytest.mark.asyncio
async def test_gap_report_boundary_and_pre_limit_period_filtering(
    session: AsyncSession, monkeypatch
):
    """Граничное время [start, now] и фильтрация до LIMIT=500 на реальной БД."""
    now = datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc)
    now_naive = now.replace(tzinfo=None)
    start_naive = now_naive - timedelta(days=1)

    _freeze_now(monkeypatch, now)

    old_ts = now_naive - timedelta(days=3)
    # 505 уникальных старых популярных и неотвеченных вопросов с count=2 —
    # без фильтрации до GROUP BY/LIMIT=500 свежий вопрос с count=1 не попал бы в топы.
    bulk: list[MessageLog] = []
    for i in range(505):
        for j in range(2):
            bulk.append(
                MessageLog(
                    user_id=1,
                    session_id=f"bulk-pop-{i}-{j}",
                    message_type="user_input",
                    content=f"Старый частый вопрос #{i}",
                    created_at=old_ts,
                )
            )
            bulk.append(
                MessageLog(
                    user_id=1,
                    session_id=f"bulk-na-{i}-{j}",
                    message_type="no_answer",
                    content=f"Старый неотвеченный #{i}",
                    created_at=old_ts,
                )
            )

    # Граница ровно start и на 1 микросекунду раньше start.
    bulk.extend(
        [
            MessageLog(
                user_id=2,
                session_id="before-start",
                message_type="user_input",
                content="На микросекунду раньше",
                created_at=start_naive - timedelta(microseconds=1),
            ),
            MessageLog(
                user_id=3,
                session_id="at-start",
                message_type="user_input",
                content="Ровно на границе start",
                created_at=start_naive,
            ),
            MessageLog(
                user_id=4,
                session_id="at-end-na",
                message_type="no_answer",
                content="Неотвеченный на границе end",
                created_at=now_naive,
            ),
            MessageLog(
                user_id=5,
                session_id="before-start-na",
                message_type="no_answer",
                content="Неотвеченный до границы",
                created_at=start_naive - timedelta(microseconds=1),
            ),
        ]
    )
    session.add_all(bulk)
    await session.commit()

    report = await gap_report_service.build_gap_report(session, days=1, limit=10)

    assert [q["question"] for q in report["popular"]] == ["Ровно на границе start"]
    assert [q["question"] for q in report["unanswered"]] == [
        "Неотвеченный на границе end"
    ]
    assert report["quality"]["questions"] == 1
