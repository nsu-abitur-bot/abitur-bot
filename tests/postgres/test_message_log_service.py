import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.models import QuestionEmbeddingCache
from db.postgres.services.message_log import MessageLogService


class FakePopularQuestionEmbeddings:
    model = "fake-embedding-model"

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "поступ" in lowered or "нужно" in lowered:
                vectors.append([1.0, 0.0])
            elif "общежит" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([-1.0, 0.0])
        return vectors


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
async def test_get_popular_questions_merges_semantically_close_questions(
    session: AsyncSession,
):
    service = MessageLogService(session)

    await service.create_log(
        user_id=1,
        session_id="session-1",
        message_type="user_input",
        content="[from alice] Как поступить в НГУ?",
    )
    await service.create_log(
        user_id=2,
        session_id="session-2",
        message_type="user_input",
        content="[from bob] Как поступить в НГУ?",
    )
    await service.create_log(
        user_id=3,
        session_id="session-3",
        message_type="user_input",
        content="Что нужно для поступления в НГУ?",
    )
    await service.create_log(
        user_id=4,
        session_id="session-4",
        message_type="user_input",
        content="Сколько стоит общежитие?",
    )

    popular = await service.get_popular_questions(
        limit=2,
        raw_limit=10,
        similarity_threshold=0.9,
        embedder=FakePopularQuestionEmbeddings(),
    )

    assert popular[0]["question"] == "Как поступить в НГУ?"
    assert popular[0]["count"] == 3
    assert popular[0]["variants"] == [
        "Как поступить в НГУ?",
        "Что нужно для поступления в НГУ?",
    ]
    assert popular[1]["question"] == "Сколько стоит общежитие?"
    assert popular[1]["count"] == 1


@pytest.mark.asyncio
async def test_get_popular_questions_reuses_cached_embeddings(
    session: AsyncSession,
):
    service = MessageLogService(session)
    embedder = FakePopularQuestionEmbeddings()

    await service.create_log(
        user_id=1,
        session_id="session-1",
        message_type="user_input",
        content="Как поступить в НГУ?",
    )

    await service.get_popular_questions(
        limit=1,
        raw_limit=10,
        embedder=embedder,
    )
    await service.get_popular_questions(
        limit=1,
        raw_limit=10,
        embedder=embedder,
    )

    cache_count = (
        await session.execute(select(func.count()).select_from(QuestionEmbeddingCache))
    ).scalar_one()

    assert embedder.calls == 1
    assert cache_count == 1
