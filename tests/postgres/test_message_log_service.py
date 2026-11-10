from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.services.message_log import MessageLogService


@pytest.mark.asyncio
async def test_get_logs_by_base_session_includes_dialog_sessions(
    session: AsyncSession,
):
    service = MessageLogService(session)

    await service.create_log(
        user_id=1,
        session_id="session-1",
        message_type="user_input",
        content="Старый вопрос",
    )
    await service.create_log(
        user_id=1,
        session_id="session-1:dialog:1",
        message_type="rag_query",
        content="Запрос к базе знаний",
    )
    await service.create_log(
        user_id=1,
        session_id="session-1:dialog:1",
        message_type="rag_response",
        content="Ответ базы знаний",
    )
    await service.create_log(
        user_id=1,
        session_id="session-2",
        message_type="user_input",
        content="Чужой вопрос",
    )

    logs = await service.get_logs_by_session("session-1")

    assert {log.session_id for log in logs} == {"session-1", "session-1:dialog:1"}
    assert {log.message_type for log in logs} == {
        "user_input",
        "rag_query",
        "rag_response",
    }


@pytest.mark.asyncio
async def test_get_logs_by_dialog_session_returns_only_that_dialog(
    session: AsyncSession,
):
    service = MessageLogService(session)

    await service.create_log(
        user_id=1,
        session_id="session-1:dialog:1",
        message_type="rag_response",
        content="Первый диалог",
    )
    await service.create_log(
        user_id=1,
        session_id="session-1:dialog:2",
        message_type="rag_response",
        content="Второй диалог",
    )

    logs = await service.get_logs_by_session("session-1:dialog:1")

    assert len(logs) == 1
    assert logs[0].content == "Первый диалог"


@pytest.mark.asyncio
async def test_get_question_groups_merges_exact_matches(session: AsyncSession):
    """Слой данных объединяет только совпадения после нормализации текста."""
    service = MessageLogService(session)

    for index, content in enumerate(
        [
            "[from alice] Как поступить в НГУ?",
            "[from bob] как поступить в нгу",
            "Сколько стоит общежитие?",
        ],
        start=1,
    ):
        await service.create_log(
            user_id=index,
            session_id=f"session-{index}",
            message_type="user_input",
            content=content,
        )

    groups = await service.get_question_groups(raw_limit=10)

    assert [group["count"] for group in groups] == [2, 1]
    first = groups[0]
    # Имя автора в префиксе не попадает ни в представителя, ни в варианты.
    assert first["question"] == "Как поступить в НГУ?"
    assert first["canonical"] == "как поступить в нгу"
    assert first["variants"] == ["Как поступить в НГУ?", "как поступить в нгу"]
    assert groups[1]["question"] == "Сколько стоит общежитие?"


@pytest.mark.asyncio
async def test_get_question_groups_filters_by_message_type(session: AsyncSession):
    """Очередь неотвеченных (#258): тип no_answer отделяет пробелы контента
    от всех вопросов и сервисных записей."""
    service = MessageLogService(session)

    entries = [
        ("user_input", "Как поступить в НГУ?"),
        ("faq_match", "Как поступить в НГУ?"),
        ("llm_response", "Документы принимают с 20 июня."),
        ("no_answer", "Есть ли стипендия для аспирантов?"),
        ("no_answer", "есть ли стипендия для аспирантов?"),
        ("no_answer", "Как проходит апелляция?"),
    ]
    for index, (message_type, content) in enumerate(entries, start=1):
        await service.create_log(
            user_id=index,
            session_id=f"session-{index}",
            message_type=message_type,
            content=content,
        )

    groups = await service.get_question_groups(raw_limit=10, message_type="no_answer")

    assert [group["count"] for group in groups] == [2, 1]
    assert groups[0]["question"] == "Есть ли стипендия для аспирантов?"
    assert groups[0]["variants"] == [
        "Есть ли стипендия для аспирантов?",
        "есть ли стипендия для аспирантов?",
    ]
    assert groups[1]["question"] == "Как проходит апелляция?"


@pytest.mark.asyncio
async def test_get_faq_hit_stats_counts_and_rate(session: AsyncSession):
    """Доля срабатываний FAQ: faq_match / user_input, прочие типы не считаются."""
    service = MessageLogService(session)

    for index in range(3):
        await service.create_log(
            user_id=1,
            session_id="session-1",
            message_type="user_input",
            content=f"Вопрос {index}",
        )
    await service.create_log(
        user_id=1,
        session_id="session-1",
        message_type="faq_match",
        content="Готовый ответ",
    )
    await service.create_log(
        user_id=1,
        session_id="session-1",
        message_type="llm_response",
        content="Ответ модели",
    )

    stats = await service.get_faq_hit_stats(start=None, end=None, group_by="day")

    assert stats["total_questions"] == 3
    assert stats["total_hits"] == 1
    assert stats["hit_rate"] == pytest.approx(1 / 3)
    assert len(stats["buckets"]) == 1
    bucket = stats["buckets"][0]
    assert bucket["questions"] == 3
    assert bucket["hits"] == 1
    assert bucket["hit_rate"] == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_get_faq_hit_stats_empty_period(session: AsyncSession):
    service = MessageLogService(session)

    stats = await service.get_faq_hit_stats(start=None, end=None, group_by="day")

    assert stats["total_questions"] == 0
    assert stats["total_hits"] == 0
    assert stats["hit_rate"] is None
    assert stats["buckets"] == []


@pytest.mark.asyncio
async def test_get_faq_hit_stats_respects_period(session: AsyncSession):
    service = MessageLogService(session)
    await service.create_log(
        user_id=1,
        session_id="session-1",
        message_type="user_input",
        content="Вопрос",
    )

    future = datetime(2100, 1, 1)
    stats = await service.get_faq_hit_stats(start=future, end=None, group_by="day")

    assert stats["total_questions"] == 0
    assert stats["hit_rate"] is None


@pytest.mark.asyncio
async def test_get_answer_stats_aggregates_latency_no_answer_errors(
    session: AsyncSession,
):
    """Метрики по периодам (#242): p50/p95 по latency_ms, доли неответов
    и ошибок; записи без метаданных участвуют только в счётчике ответов."""
    from db.postgres.models import MessageLog

    service = MessageLogService(session)
    day_one = datetime(2026, 11, 8, 12, 0, 0)
    day_two = datetime(2026, 11, 9, 12, 0, 0)
    rows = [
        MessageLog(
            user_id=1,
            session_id="s",
            message_type="user_input",
            content="В1",
            created_at=day_one,
        ),
        MessageLog(
            user_id=1,
            session_id="s",
            message_type="user_input",
            content="В2",
            created_at=day_one,
        ),
        MessageLog(
            user_id=1,
            session_id="s",
            message_type="llm_response",
            content="Не нашёл информации",
            created_at=day_one,
            message_metadata={"latency_ms": 100.0, "no_answer": True},
        ),
        MessageLog(
            user_id=1,
            session_id="s",
            message_type="llm_response",
            content="Ответ",
            created_at=day_one,
            message_metadata={"latency_ms": 300.0, "no_answer": False},
        ),
        MessageLog(
            user_id=1,
            session_id="s",
            message_type="llm_response",
            content="Сбой",
            created_at=day_one,
            message_metadata={"latency_ms": 500.0, "error": True},
        ),
        MessageLog(
            user_id=1,
            session_id="s",
            message_type="user_input",
            content="В3",
            created_at=day_two,
        ),
        MessageLog(
            user_id=1,
            session_id="s",
            message_type="llm_response",
            content="Старая запись без метрик",
            created_at=day_two,
        ),
        MessageLog(
            user_id=1,
            session_id="s",
            message_type="faq_match",
            content="Готовый ответ",
            created_at=day_two,
        ),
    ]
    session.add_all(rows)
    await session.commit()

    stats = await service.get_answer_stats(
        start=datetime(2026, 11, 8), end=datetime(2026, 11, 9, 23, 59), group_by="day"
    )

    assert stats["total_questions"] == 3
    assert stats["total_answers"] == 4
    assert stats["total_no_answer"] == 1
    assert stats["no_answer_share"] == pytest.approx(1 / 3)
    assert stats["total_errors"] == 1
    assert stats["error_share"] == pytest.approx(1 / 3)

    first, second = stats["buckets"]
    assert first["period"] == datetime(2026, 11, 8)
    assert (first["questions"], first["answers"]) == (2, 3)
    assert first["no_answer_share"] == pytest.approx(0.5)
    assert first["error_share"] == pytest.approx(0.5)
    # percentile_cont по {100, 300, 500}: p50 = 300, p95 интерполируется в 480.
    assert first["latency_ms_p50"] == pytest.approx(300.0)
    assert first["latency_ms_p95"] == pytest.approx(480.0)

    # Запись без метрик считается ответом, но в перцентили не попадает.
    assert (second["questions"], second["answers"]) == (1, 1)
    assert second["no_answer"] == 0
    assert second["errors"] == 0
    assert second["latency_ms_p50"] is None
    assert second["latency_ms_p95"] is None
