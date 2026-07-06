"""Тесты CRAG-фильтра.

Используют реальную тестовую БД (конвенция проекта: без моков для БД) для
авторитетной таблицы факультетов. LLM-грейдер детерминированно подменяется,
чтобы тесты были стабильны и не ходили в сеть.
"""

import os
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from db.postgres.models import Base
from db.postgres.services.faculty import FacultyService

load_dotenv()

TEST_DB_NAME = os.getenv("TEST_DB_NAME", "abitur_test")
_db_host = os.getenv("DB_HOST", "localhost")
_db_port = os.getenv("DB_PORT", "5432")
_db_user = os.getenv("DB_USER", "postgres")
_db_password = os.getenv("DB_PASSWORD", "postgres")

TEST_DATABASE_URL = (
    f"postgresql+asyncpg://{_db_user}:{_db_password}@{_db_host}:{_db_port}/{TEST_DB_NAME}"
)


@pytest_asyncio.fixture(scope="function")
async def crag_engine() -> AsyncGenerator[AsyncEngine, None]:
    engine = create_async_engine(TEST_DATABASE_URL, echo=False, pool_pre_ping=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def crag_sessionmaker(crag_engine: AsyncEngine):
    return async_sessionmaker(
        bind=crag_engine, class_=AsyncSession, expire_on_commit=False
    )


@pytest_asyncio.fixture
async def seeded(crag_sessionmaker, monkeypatch) -> async_sessionmaker:
    """Сидит ФИТ и экономфак и подменяет AsyncSessionLocal внутри CRAG."""
    import rag.crag as crag_module

    async with crag_sessionmaker() as session:
        await FacultyService(session).upsert_from_seed(
            [
                {
                    "name": "Факультет информационных технологий",
                    "aliases": ["ФИТ"],
                    "programs": [
                        {
                            "name": "Программная инженерия",
                            "code": "09.03.04",
                            "level": "bachelor",
                        }
                    ],
                },
                {
                    "name": "Экономический факультет",
                    "aliases": ["ЭФ"],
                    "programs": [
                        {
                            "name": "Бизнес-информатика",
                            "code": "38.03.05",
                            "level": "bachelor",
                        }
                    ],
                },
            ]
        )

    monkeypatch.setattr(crag_module, "AsyncSessionLocal", crag_sessionmaker)
    return crag_sessionmaker


@pytest.mark.asyncio
async def test_detect_faculty_hint(seeded):
    from rag.crag import detect_faculty_hint

    hint = await detect_faculty_hint("проходные баллы на ФИТ")
    assert hint.faculty is not None
    assert hint.faculty.name == "Факультет информационных технологий"
    assert hint.level is None

    hint_master = await detect_faculty_hint("магистратура на ФИТ")
    assert hint_master.level == "master"


@pytest.mark.asyncio
async def test_fit_question_drops_foreign_faculty_program(seeded, monkeypatch):
    """Ключевой кейс: вопрос про ФИТ не должен оставлять бизнес-информатику.

    Бизнес-информатика принадлежит экономфаку, поэтому авторитетная фильтрация
    обязана отсечь чанк, даже если LLM-грейдер пропустил бы его.
    """
    import rag.crag as crag_module
    from rag.crag import CragChunk, CragConfig, filter_chunks

    # Грейдер пропускает всё — проверяем именно авторитетную таблицу.
    async def _always_keep(question, hint, chunk, config):
        return True

    monkeypatch.setattr(crag_module, "grade_chunk", _always_keep)

    chunks = [
        CragChunk(
            index=0,
            content=("Направление Программная инженерия на ФИТ, проходной балл 250."),
            source_url="https://nsu.ru/fit",
            file_path="https://nsu.ru/fit",
        ),
        CragChunk(
            index=1,
            content=("Бизнес-информатика: проходной балл 240, обучение 4 года."),
            source_url="https://nsu.ru/econ",
            file_path="https://nsu.ru/econ",
        ),
    ]

    config = CragConfig(
        enabled=True,
        relevance_threshold=0.5,
        min_chunks=1,
        allow_refine=False,
        use_faculty_table=True,
        max_graded_chunks=12,
    )

    kept, hint = await filter_chunks("проходные баллы на ФИТ", chunks, config)

    assert hint.faculty is not None
    kept_contents = " ".join(c.content for c in kept).lower()
    assert "программная инженерия" in kept_contents
    assert "бизнес-информатика" not in kept_contents


@pytest.mark.asyncio
async def test_mixed_chunk_scrubs_foreign_keeps_own(seeded, monkeypatch):
    """Смешанный чанк: чужое направление вычищается, своё остаётся.

    Боевые чанки — плоские списки бюджетных мест по всем факультетам. Фильтр
    должен вырезать сегмент про бизнес-информатику, но НЕ терять данные ФИТ из
    того же чанка (сентенс-левел, а не отбрасывание чанка целиком).
    """
    import rag.crag as crag_module
    from rag.crag import CragChunk, CragConfig, filter_chunks

    async def _always_keep(question, hint, chunk, config):
        return True

    monkeypatch.setattr(crag_module, "grade_chunk", _always_keep)

    mixed = CragChunk(
        index=0,
        content=(
            "Для направления подготовки «Программная инженерия (09.03.04, "
            "бакалавр)» количество бюджетных мест всего: 125.\n\n"
            "Для направления подготовки «Бизнес-информатика (38.03.05, "
            "бакалавр)» количество бюджетных мест всего: 30."
        ),
        source_url="https://nsu.ru/places",
        file_path="https://nsu.ru/places",
    )

    config = CragConfig(
        enabled=True,
        relevance_threshold=0.5,
        min_chunks=1,
        allow_refine=False,
        use_faculty_table=True,
        max_graded_chunks=12,
    )

    kept, _ = await filter_chunks("проходные баллы на ФИТ", [mixed], config)

    # Чанк не выброшен целиком — данные ФИТ сохранены.
    assert len(kept) == 1
    content = kept[0].content.lower()
    assert "программная инженерия" in content
    assert "125" in content
    # Чужое направление вычищено.
    assert "бизнес-информатика" not in content


@pytest.mark.asyncio
async def test_scrub_is_format_independent(seeded, monkeypatch):
    """Фильтр не зависит от конкретной формулировки документа.

    Даже если после переразбора формат другой (без маркера «Для направления
    подготовки», одной строкой), чужое направление вычищается, а своё остаётся —
    с сохранением названия целиком (точка внутри названия не рвёт его).
    """
    import rag.crag as crag_module
    from rag.crag import CragChunk, CragConfig, filter_chunks

    async def _always_keep(question, hint, chunk, config):
        return True

    monkeypatch.setattr(crag_module, "grade_chunk", _always_keep)

    reformatted = CragChunk(
        index=0,
        content=(
            "«Бизнес-информатика (38.03.05, бакалавр)» — 30 бюджетных мест. "
            "«Программная инженерия (09.03.04, бакалавр)» — 125 бюджетных мест."
        ),
        source_url="https://nsu.ru/x",
        file_path="https://nsu.ru/x",
    )

    config = CragConfig(
        enabled=True,
        relevance_threshold=0.5,
        min_chunks=1,
        allow_refine=False,
        use_faculty_table=True,
        max_graded_chunks=12,
    )

    kept, _ = await filter_chunks("проходные баллы на ФИТ", [reformatted], config)

    assert len(kept) == 1
    content = kept[0].content.lower()
    assert "программная инженерия" in content
    assert "125" in content
    assert "бизнес-информатика" not in content


@pytest.mark.asyncio
async def test_grader_drops_irrelevant(seeded, monkeypatch):
    """Если LLM-грейдер признал чанк нерелевантным — он отсекается."""
    import rag.crag as crag_module
    from rag.crag import CragChunk, CragConfig, filter_chunks

    async def _fake_generate(messages, profile=None, **kwargs):
        # Решаем по содержимому фрагмента.
        user = messages[-1].content.lower()
        if "общежит" in user:
            return '{"score": 0.1, "relevant": false}'
        return '{"score": 0.9, "relevant": true}'

    class _FakeProvider:
        async def generate(self, messages, profile=None, **kwargs):
            return await _fake_generate(messages, profile=profile, **kwargs)

    monkeypatch.setattr(crag_module, "get_crag_config", lambda: CragConfig())
    monkeypatch.setattr("llm.factory.get_llm_provider", lambda: _FakeProvider())

    chunks = [
        CragChunk(
            index=0,
            content="Программная инженерия: проходной балл 250.",
            source_url="https://nsu.ru/fit",
            file_path="https://nsu.ru/fit",
        ),
        CragChunk(
            index=1,
            content="Общежитие НГУ находится в Академгородке.",
            source_url="https://nsu.ru/dorm",
            file_path="https://nsu.ru/dorm",
        ),
    ]

    config = CragConfig(
        enabled=True,
        relevance_threshold=0.5,
        min_chunks=1,
        allow_refine=False,
        use_faculty_table=False,
        max_graded_chunks=12,
    )

    kept, _ = await filter_chunks("проходные баллы на ФИТ", chunks, config)
    kept_contents = " ".join(c.content for c in kept).lower()
    assert "программная инженерия" in kept_contents
    assert "общежит" not in kept_contents


@pytest.mark.asyncio
async def test_disabled_crag_keeps_all(seeded, monkeypatch):
    """use_faculty_table=False и пропускающий грейдер → ничего не теряем."""
    import rag.crag as crag_module
    from rag.crag import CragChunk, CragConfig, filter_chunks

    async def _always_keep(question, hint, chunk, config):
        return True

    monkeypatch.setattr(crag_module, "grade_chunk", _always_keep)

    chunks = [
        CragChunk(index=0, content="A", source_url=None, file_path=None),
        CragChunk(index=1, content="B", source_url=None, file_path=None),
    ]
    config = CragConfig(use_faculty_table=False)
    kept, _ = await filter_chunks("вопрос без факультета", chunks, config)
    assert len(kept) == 2
