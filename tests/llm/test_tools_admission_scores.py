"""Тесты инструмента get_admission_scores против реальной тестовой БД.

Проверяем, что для двух направлений ФИТ инструмент отдаёт РАЗНЫЕ проходные
баллы, а также работу метрик (проходной/средний) и пустого результата.
"""

import os

import pytest
import pytest_asyncio
from db.postgres.services.admission_score import AdmissionScoreService, ScoreRow
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from db.postgres.models import Base
from db.postgres.services.faculty import FacultyService
from llm.tools import admission_scores as tool_mod
from llm.tools import default_tool_executor
from llm.tools.admission_scores import execute_admission_scores

TEST_DB_NAME = os.getenv("TEST_DB_NAME", "abitur_test")
_DB_HOST = os.getenv("DB_HOST", "localhost")
_DB_PORT = os.getenv("DB_PORT", "5432")
_DB_USER = os.getenv("DB_USER", "postgres")
_DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

TEST_DATABASE_URL = (
    f"postgresql+asyncpg://{_DB_USER}:{_DB_PASSWORD}@{_DB_HOST}:{_DB_PORT}/{TEST_DB_NAME}"
)


@pytest_asyncio.fixture
async def sessionmaker_bound(monkeypatch):
    """Создаёт тестовую БД и подменяет AsyncSessionLocal в модуле инструмента."""
    engine = create_async_engine(
        TEST_DATABASE_URL, echo=False, pool_size=1, max_overflow=0, pool_pre_ping=True
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    maker = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(tool_mod, "AsyncSessionLocal", maker)

    yield maker

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def _seed_fit(maker) -> None:
    async with maker() as session:
        faculty_service = FacultyService(session)
        await faculty_service.upsert_from_seed(
            [
                {
                    "name": "Факультет информационных технологий",
                    "aliases": ["ФИТ"],
                    "programs": [
                        {
                            "name": "Программная инженерия и компьютерные науки",
                            "code": "09.03.04",
                            "level": "bachelor",
                        },
                        {
                            "name": "Компьютерные науки и системотехника",
                            "code": "09.03.01",
                            "level": "bachelor",
                        },
                    ],
                }
            ]
        )
        score_service = AdmissionScoreService(session)
        await score_service.upsert_from_rows(
            [
                ScoreRow(
                    faculty_name="ФИТ",
                    program_name="Программная инженерия и компьютерные науки",
                    code="09.03.04",
                    year=2024,
                    form="budget",
                    passing_score=246,
                    average_score=84.2,
                ),
                ScoreRow(
                    faculty_name="ФИТ",
                    program_name="Компьютерные науки и системотехника",
                    code="09.03.01",
                    year=2024,
                    form="budget",
                    passing_score=260,
                    average_score=88.1,
                ),
            ]
        )


@pytest.mark.asyncio
async def test_execute_returns_distinct_passing_scores(sessionmaker_bound):
    await _seed_fit(sessionmaker_bound)

    text = await execute_admission_scores({"faculty": "ФИТ", "year": 2024})

    # Оба направления присутствуют с РАЗНЫМИ проходными баллами.
    assert "246" in text
    assert "260" in text
    assert "Программная инженерия и компьютерные науки" in text
    assert "Компьютерные науки и системотехника" in text
    # По умолчанию — только проходной, без среднего.
    assert "84.2" not in text
    assert "88.1" not in text


@pytest.mark.asyncio
async def test_execute_includes_average_when_requested(sessionmaker_bound):
    await _seed_fit(sessionmaker_bound)

    text = await execute_admission_scores(
        {"faculty": "ФИТ", "year": 2024, "metric": "both"}
    )
    assert "246" in text
    assert "84.2" in text
    assert "средний" in text


@pytest.mark.asyncio
async def test_execute_empty_returns_marker(sessionmaker_bound):
    await _seed_fit(sessionmaker_bound)

    text = await execute_admission_scores({"faculty": "Несуществующий"})
    assert text == "По этому запросу данных о проходных баллах нет."


@pytest.mark.asyncio
async def test_default_tool_executor_dispatches(sessionmaker_bound):
    await _seed_fit(sessionmaker_bound)

    text = await default_tool_executor(
        "get_admission_scores", {"faculty": "ФИТ", "year": 2024}
    )
    assert "246" in text and "260" in text


@pytest.mark.asyncio
async def test_default_tool_executor_unknown_tool():
    text = await default_tool_executor("does_not_exist", {})
    assert "does_not_exist" in text
