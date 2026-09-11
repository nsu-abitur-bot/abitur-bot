"""Кластеризация популярных вопросов (api/services/popular_questions.py).

Тесты живут в tests/postgres/, потому что кэш векторов лежит в реальной базе —
см. соглашение в docs/agents/testing.md. Сама кластеризация принадлежит слою
api: сервису БД она недоступна, иначе получается кольцевой импорт db <-> llm.
"""

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.services.popular_questions import get_popular_questions
from db.postgres.models import QuestionEmbeddingCache
from db.postgres.services.message_log import MessageLogService


class FakeEmbeddings:
    """Двумерные векторы: «поступление» и «общежитие» — разные направления."""

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


class BrokenEmbeddings:
    """Возвращает меньше векторов, чем запросили."""

    model = "broken-embedding-model"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0]]


async def _log_questions(service: MessageLogService, questions: list[str]) -> None:
    for index, content in enumerate(questions, start=1):
        await service.create_log(
            user_id=index,
            session_id=f"session-{index}",
            message_type="user_input",
            content=content,
        )


@pytest.mark.asyncio
async def test_merges_semantically_close_questions(session: AsyncSession):
    service = MessageLogService(session)
    await _log_questions(
        service,
        [
            "[from alice] Как поступить в НГУ?",
            "[from bob] Как поступить в НГУ?",
            "Что нужно для поступления в НГУ?",
            "Сколько стоит общежитие?",
        ],
    )

    popular = await get_popular_questions(
        service,
        limit=2,
        raw_limit=10,
        similarity_threshold=0.9,
        embedder=FakeEmbeddings(),
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
async def test_reuses_cached_embeddings(session: AsyncSession):
    service = MessageLogService(session)
    embedder = FakeEmbeddings()
    await _log_questions(service, ["Как поступить в НГУ?"])

    await get_popular_questions(service, limit=1, raw_limit=10, embedder=embedder)
    await get_popular_questions(service, limit=1, raw_limit=10, embedder=embedder)

    cache_count = (
        await session.execute(select(func.count()).select_from(QuestionEmbeddingCache))
    ).scalar_one()

    assert embedder.calls == 1
    assert cache_count == 1


@pytest.mark.asyncio
async def test_no_questions_gives_empty_list(session: AsyncSession):
    service = MessageLogService(session)
    assert await get_popular_questions(service, embedder=FakeEmbeddings()) == []


@pytest.mark.asyncio
async def test_broken_embedder_falls_back_to_exact_groups(session: AsyncSession):
    """Сбой эмбеддингов не должен ронять эндпоинт — отвечаем точными группами."""
    service = MessageLogService(session)
    await _log_questions(
        service,
        [
            "Как поступить в НГУ?",
            "Как поступить в НГУ?",
            "Что нужно для поступления в НГУ?",
        ],
    )

    popular = await get_popular_questions(
        service,
        limit=5,
        raw_limit=10,
        embedder=BrokenEmbeddings(),
    )

    # Близкие формулировки не слились, но точные совпадения объединены.
    assert [item["question"] for item in popular] == [
        "Как поступить в НГУ?",
        "Что нужно для поступления в НГУ?",
    ]
    assert popular[0]["count"] == 2
