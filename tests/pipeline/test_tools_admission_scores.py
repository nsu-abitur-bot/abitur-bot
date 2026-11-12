"""Тесты инструментов баллов и справочника против реальной тестовой БД.

Проверяем, что для двух направлений ФИТ инструмент отдаёт РАЗНЫЕ проходные
баллы, а также работу метрик (проходной/средний) и пустого результата.
"""

import json
import os
from types import SimpleNamespace
from typing import cast

import pytest
import pytest_asyncio
from lightrag import LightRAG
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abbrev.expander import AbbrevExpander
from db.postgres.models import Base
from db.postgres.services.admission_score import AdmissionScoreService, ScoreRow
from db.postgres.services.faculty import FacultyService
from pipeline.tools import admission_scores as tool_mod
from pipeline.tools import default_tool_executor
from pipeline.tools.admission_scores import (
    AdmissionScoresResult,
    execute_admission_scores,
)
from rag.program_exams import retrieve_program_exams
from rag.trace import LightragTrace, RetrievalResult

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


@pytest.mark.parametrize("use_crag", [True, False], ids=["crag", "lightrag"])
@pytest.mark.parametrize(
    ("question", "metric", "expected"),
    [
        (
            "Какой проходной балл ЕГЭ на 09.03.01 за 2024 год?",
            "passing",
            "проходной 260",
        ),
        ("Какой средний балл ЕГЭ на 09.03.01?", "average", "средний 88.1"),
    ],
)
async def test_ege_score_question_uses_sql_without_program_card(
    sessionmaker_bound, monkeypatch, use_crag, question, metric, expected
):
    from llm.base import LLMResult
    from pipeline import llm_client

    await _seed_fit(sessionmaker_bound)

    async def crag_config():
        return SimpleNamespace(enabled=use_crag)

    async def retrieve(
        query: str,
        conversation_history: str | None = None,
        program_question: str | None = None,
        **_,
    ) -> RetrievalResult:
        result = await retrieve_program_exams(
            cast(LightRAG, None),
            program_question or query,
            [],
            LightragTrace(),
            3,
            conversation_history,
        )
        return result or RetrievalResult(context="")

    class Provider:
        async def generate_with_usage(self, messages, profile):
            payload = json.loads(messages[-1].content)
            [score] = payload["admission_scores"]
            return LLMResult(
                text=json.dumps({"verdict": "supported", "quotes": [score["result"]]})
            )

        async def generate_with_tools(self, messages, **kwargs):
            result = await kwargs["tool_executor"](
                "get_admission_scores",
                {"program": "09.03.01", "metric": metric, "year": 2024},
            )
            return LLMResult(text=str(result))

    monkeypatch.setattr(llm_client, "load_crag_config", crag_config)
    monkeypatch.setattr(llm_client, "query_graph_with_crag", retrieve)
    monkeypatch.setattr(llm_client, "query_graph_with_sources", retrieve)
    monkeypatch.setattr(llm_client, "get_llm_provider", Provider)

    ctx = llm_client._Ctx(session_id="ege-score", user_id=0)
    rag = await llm_client._retrieve_context(question, "", ctx)
    answer, _ = await llm_client._generate_answer(rag, [], question, ctx)

    assert expected in answer
    assert ctx.trace.tools_called == ["get_admission_scores"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "profile",
        "code_group",
        "empty_paid",
        "renamed",
        "same_code",
        "reference",
        "named_same_code",
        "same_named_code",
        "same_named",
        "narrow_group_name",
    ],
)
async def test_shared_code_cannot_substitute_a_different_profile(
    sessionmaker_bound, expander, monkeypatch, mode
):
    from langchain_core.messages import AIMessage, HumanMessage

    from llm.base import LLMResult
    from pipeline import llm_client

    faculty = "Тестовый факультет"
    first = "Информатика. Программная инженерия"
    second = "Информатика. Компьютерные науки"
    async with sessionmaker_bound() as session:
        await FacultyService(session).upsert_from_seed(
            [
                {
                    "name": faculty,
                    "programs": [
                        {"name": first, "code": "09.03.01", "level": "bachelor"},
                        {"name": second, "code": "09.03.01", "level": "bachelor"},
                    ],
                }
            ]
        )
        await AdmissionScoreService(session).upsert_from_rows(
            [
                ScoreRow(faculty, first, "09.03.01", 2024, "budget", 141, 63.27),
                ScoreRow(faculty, second, "09.03.01", 2024, "budget", 260, 88.1),
                ScoreRow(faculty, second, "09.03.01", 2024, "paid", 200, 99),
            ]
        )
    initial_program = "09.03.01" if mode in ("code_group", "narrow_group_name") else first
    initial = await execute_admission_scores(
        {
            "faculty": faculty,
            "program": initial_program,
            "year": 2024,
            "form": "budget",
        }
    )
    assert isinstance(initial, AdmissionScoresResult)
    if mode == "renamed":
        from db.postgres.models import Program

        async with sessionmaker_bound() as session:
            program = await session.get(Program, initial.scope["_program_ids"][0])
            assert program is not None
            program.name = "Новое название первой программы"
            program.code = "09.03.02"
            await session.commit()
    calls, published = [], []
    form = "paid" if mode == "empty_paid" else "budget"

    class Provider:
        async def generate_with_tools(self, messages, **kwargs):
            calls.append(kwargs)
            args = {
                "faculty": faculty,
                "program": second,
                "year": 2024,
                "form": form,
                "metric": "average",
                "level": "bachelor",
            }
            if len(calls) > 1:
                ids = kwargs["tools"][0].parameters["properties"]["program_ids"]["items"][
                    "enum"
                ]
                expected_ids = set(initial.scope["_program_ids"])
                if mode == "narrow_group_name":
                    expected_ids = {
                        key
                        for key, names in initial.scope["_program_names"].items()
                        if first in names
                    }
                assert set(ids) == expected_ids
                args = {
                    "program_ids": ids,
                    "year": 2024,
                    "form": form,
                    "metric": "average",
                    "level": "bachelor",
                }
            output = await kwargs["tool_executor"]("get_admission_scores", args)
            return LLMResult(str(output))

    async def publish(text):
        published.append(text)

    monkeypatch.setattr(llm_client, "get_llm_provider", Provider)
    question = {
        "empty_paid": "А средний балл на платное?",
        "same_code": "А средний балл на 09.03.01?",
        "reference": "А средний балл по нему?",
        "named_same_code": f"А средний балл на {second} (09.03.01)?",
        "same_named_code": f"А средний балл на {first} (09.03.01)?",
        "same_named": f"А средний балл на {first}?",
        "narrow_group_name": f"А средний балл на {first} (09.03.01)?",
    }.get(mode, "А средний?")
    text, _, _ = await llm_client._run_tool_loop(
        [
            HumanMessage(content=f"Какой проходной балл на {initial_program}?"),
            AIMessage(
                content=str(initial),
                additional_kwargs={"admission_scores": initial.scope},
            ),
            HumanMessage(content=question),
        ],
        "shared-code-regression",
        stream_callback=publish,
    )
    if mode == "named_same_code":
        assert "88.1" in text and "63.27" not in text
        assert len(calls) == 1
        assert published == [text]
        return
    if mode == "empty_paid":
        assert text == "По этому запросу данных о проходных баллах нет."
        assert "99" not in text
    else:
        assert "63.27" in text
        assert ("88.1" in text) == (mode == "code_group")
    assert len(calls) == 2
    assert published == [text]


@pytest.mark.asyncio
async def test_program_name_resolves_master_clarification(
    sessionmaker_bound, expander, monkeypatch
):
    from langchain_core.messages import AIMessage, HumanMessage

    from llm.base import LLMResult
    from pipeline import llm_client

    await _seed_fit(sessionmaker_bound)
    initial = await execute_admission_scores({"program": "09.03.01"})
    assert isinstance(initial, AdmissionScoresResult)
    async with sessionmaker_bound() as session:
        await FacultyService(session).upsert_from_seed(
            [
                {
                    "name": "Магистерский факультет",
                    "programs": [
                        {"name": "Анализ данных", "code": "09.04.01", "level": "master"}
                    ],
                }
            ]
        )
        await AdmissionScoreService(session).upsert_from_rows(
            [
                ScoreRow(
                    "Магистерский факультет",
                    "Анализ данных",
                    "09.04.01",
                    2024,
                    "budget",
                    80,
                    85,
                    level="master",
                )
            ]
        )
    calls = []

    class Provider:
        async def generate_with_tools(self, messages, **kwargs):
            calls.append(1)
            result = await kwargs["tool_executor"](
                "get_admission_scores",
                {"program": "Анализ данных", "level": "master", "metric": "average"},
            )
            return LLMResult(str(result))

    monkeypatch.setattr(llm_client, "get_llm_provider", Provider)
    text, _, _ = await llm_client._run_tool_loop(
        [
            AIMessage(
                content=str(initial),
                additional_kwargs={"admission_scores": initial.scope},
            ),
            HumanMessage(content="А на магистратуру?"),
            AIMessage(content=llm_client.SCORES_CLARIFY),
            HumanMessage(content="Какой средний балл на Анализ данных?"),
        ],
        "named-master-regression",
    )
    assert "Анализ данных" in text and "средний 85" in text
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "key", "bad_value", "broad_args", "values"),
    [
        (
            "А средний балл за 2023 и 2024 годы?",
            "year",
            2025,
            {"form": "budget"},
            [2023, 2024],
        ),
        (
            "А средний балл на бюджет и платное?",
            "form",
            "budget",
            {"year": 2024},
            ["budget", "paid"],
        ),
    ],
)
async def test_comparison_recovers_full_sql_query_and_preserves_context(
    sessionmaker_bound,
    expander,
    monkeypatch,
    question,
    key,
    bad_value,
    broad_args,
    values,
):
    from fakeredis.aioredis import FakeRedis

    from llm.base import LLMResult
    from pipeline import llm_client
    from pipeline.score_guard import ScoreGuard

    await _seed_fit(sessionmaker_bound)
    async with sessionmaker_bound() as session:
        await AdmissionScoreService(session).upsert_from_rows(
            [
                ScoreRow(
                    "ФИТ",
                    "Программная инженерия и компьютерные науки",
                    "09.03.04",
                    2024,
                    "paid",
                    120,
                    40,
                ),
            ]
        )
    args = {"program": "09.03.04", "metric": "average", **broad_args}
    calls = []

    class Provider:
        async def generate_with_tools(self, messages, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                await kwargs["tool_executor"](
                    "get_admission_scores", {**args, key: bad_value}
                )
                return LLMResult("Непроверенный ответ")
            assert key not in kwargs["tools"][0].parameters["properties"]
            output = await kwargs["tool_executor"]("get_admission_scores", args)
            assert isinstance(output, AdmissionScoresResult) and output.status == "ok"
            assert "84.2" in output
            if key == "year":
                assert output.scope["year"] == 2024
            else:
                assert "40" in output
            return LLMResult(str(output))

    monkeypatch.setattr(llm_client, "get_llm_provider", Provider)
    monkeypatch.setattr(llm_client, "_spawn_bg", lambda task: task.close())
    ctx = llm_client._Ctx(session_id="comparison-sql", user_id=0)
    text, _ = await llm_client._generate_answer(
        llm_client._RagResult(context="Контекст"),
        [
            {
                "role": "assistant",
                "content": "Проходной 246",
                "admission_scores": {
                    "program": "09.03.04",
                    "year": 2024,
                    "form": "budget",
                    "metric": "passing",
                },
            }
        ],
        question,
        ctx,
    )
    assert len(calls) == 2
    assert ctx.trace.tools_called == ["get_admission_scores"]
    assert ctx.score_query is not None
    assert ctx.score_query["_comparisons"] == {key: values}
    tasks = []
    history = object.__new__(llm_client.RedisClient)
    history.client = FakeRedis(decode_responses=True)
    monkeypatch.setattr(llm_client, "_spawn_bg", tasks.append)
    llm_client._store_answer(text, question, history, ctx)
    for task in tasks:
        await task
    entries = await history.get_history(ctx.session_id)
    await history.client.aclose()
    guard = ScoreGuard.from_messages(
        llm_client._build_messages("", entries, "Какой средний балл?")
    )
    assert guard is not None and guard.comparisons == {key: values}
    assert not guard.accepts({**args, key: bad_value})
    assert guard.accepts(args)
    followup = "А в 2024?" if key == "year" else "А по платному?"
    single = ScoreGuard.from_messages(llm_client._build_messages("", entries, followup))
    assert single is not None and key not in single.comparisons
    assert single.accepts({**args, key: 2024 if key == "year" else "paid"})


@pytest.mark.asyncio
async def test_master_program_after_clarification_uses_new_code(
    sessionmaker_bound, expander, monkeypatch
):
    from llm.base import LLMResult
    from pipeline import llm_client

    async with sessionmaker_bound() as session:
        await FacultyService(session).upsert_from_seed(
            [
                {
                    "name": "Тестовый факультет",
                    "programs": [
                        {
                            "name": "Тестовая магистратура",
                            "code": "09.04.01",
                            "level": "master",
                        }
                    ],
                }
            ]
        )
        await AdmissionScoreService(session).upsert_from_rows(
            [
                ScoreRow(
                    "Тестовый факультет",
                    "Тестовая магистратура",
                    "09.04.01",
                    2024,
                    "budget",
                    80,
                    85,
                    level="master",
                )
            ]
        )

    class Provider:
        async def generate_with_tools(self, messages, **kwargs):
            output = await kwargs["tool_executor"](
                "get_admission_scores",
                {
                    "program": "09.04.01",
                    "level": "master",
                    "year": 2024,
                    "form": "budget",
                },
            )
            assert isinstance(output, AdmissionScoresResult) and output.status == "ok"
            return LLMResult(str(output))

    monkeypatch.setattr(llm_client, "get_llm_provider", Provider)
    from langchain_core.messages import AIMessage, HumanMessage

    text, _, _ = await llm_client._run_tool_loop(
        [
            AIMessage(
                content="141",
                additional_kwargs={
                    "admission_scores": {
                        "program": "09.03.01",
                        "level": "bachelor",
                        "metric": "passing",
                    }
                },
            ),
            HumanMessage(content="А на магистратуру?"),
            AIMessage(content=llm_client.SCORES_CLARIFY),
            HumanMessage(content="А на 09.04.01?"),
        ],
        "master-program-sql",
    )
    assert "проходной 80" in text and "09.04.01" in text


@pytest.mark.asyncio
async def test_execute_returns_distinct_passing_scores(sessionmaker_bound):
    await _seed_fit(sessionmaker_bound)

    text = await execute_admission_scores({"faculty": "ФИТ", "year": 2024})

    # Оба направления присутствуют с РАЗНЫМИ проходными баллами.
    assert "246" in text
    assert "260" in text
    assert text.startswith("Проходные баллы прошлых лет (НГУ):")
    assert "Программная инженерия и компьютерные науки" in text
    assert "Компьютерные науки и системотехника" in text
    # По умолчанию — только проходной, без среднего.
    assert "84.2" not in text
    assert "88.1" not in text
    assert isinstance(text, AdmissionScoresResult) and text.status == "ok"


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
async def test_followup_guard_recovers_from_false_absence_using_sql(
    sessionmaker_bound, expander, monkeypatch
):
    from langchain_core.messages import AIMessage, HumanMessage

    from llm.base import LLMResult
    from pipeline import llm_client

    await _seed_fit(sessionmaker_bound)
    attempts = []
    published = []

    class Provider:
        async def generate_with_tools(self, messages, **kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                return LLMResult("Среднего балла в базе нет.")
            assert kwargs["required_tool"] == "get_admission_scores"
            output = await kwargs["tool_executor"](
                "get_admission_scores", {"program": "09.03.04", "metric": "average"}
            )
            assert isinstance(output, AdmissionScoresResult)
            assert output.status == "ok"
            return LLMResult(str(output))

    async def publish(text):
        published.append(text)

    monkeypatch.setattr(llm_client, "get_llm_provider", Provider)
    text, _, _ = await llm_client._run_tool_loop(
        [
            HumanMessage(content="Какой проходной балл на 09.03.04?"),
            AIMessage(content="2024, бюджет: проходной балл — 246."),
            HumanMessage(content="А средний балл?"),
        ],
        "sql-guard-regression",
        stream_callback=publish,
    )
    assert "84.2" in text and "88.1" not in text
    assert published == [text]
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_sql_scope_survives_storing_and_rebuilding_history(
    sessionmaker_bound, expander, monkeypatch
):
    from fakeredis.aioredis import FakeRedis

    from pipeline import llm_client
    from pipeline.score_guard import ScoreGuard

    await _seed_fit(sessionmaker_bound)
    ctx = llm_client._Ctx(session_id="scope-roundtrip", user_id=0)
    text = await ctx.execute_tool(
        "get_admission_scores", {"program": "ПИиКН", "metric": "passing"}
    )
    assert ctx.score_query is not None
    assert ctx.score_query["_program_ids"]
    assert {
        key: value for key, value in ctx.score_query.items() if not key.startswith("_")
    } == {
        "program": "09.03.04",
        "faculty": "Факультет информационных технологий",
        "level": "bachelor",
        "year": 2024,
        "form": "budget",
        "metric": "passing",
    }
    tasks = []
    history = object.__new__(llm_client.RedisClient)
    history.client = FakeRedis(decode_responses=True)
    monkeypatch.setattr(llm_client, "_spawn_bg", tasks.append)
    llm_client._store_answer(text, "Проходной балл на ПИиКН?", history, ctx)
    for task in tasks:
        await task
    entries = await history.get_history(ctx.session_id)
    await history.client.aclose()
    guard = ScoreGuard.from_messages(
        llm_client._build_messages("", entries, "а средний?")
    )
    assert guard is not None
    assert ctx.score_query is not None
    assert guard.program_ids == frozenset(ctx.score_query["_program_ids"])
    assert guard.program_names == ctx.score_query["_program_names"]
    assert guard.filters == {
        **{
            key: value
            for key, value in ctx.score_query.items()
            if not key.startswith("_")
        },
        "metric": "average",
    }


@pytest.mark.asyncio
async def test_execute_empty_returns_marker(sessionmaker_bound):
    await _seed_fit(sessionmaker_bound)

    text = await execute_admission_scores({"faculty": "Несуществующий"})
    assert text == "По этому запросу данных о проходных баллах нет."
    assert isinstance(text, AdmissionScoresResult) and text.status == "empty"


@pytest.fixture
def expander(monkeypatch) -> AbbrevExpander:
    """Свой словарь аббревиатур — как будто его завели в админке."""
    exp = AbbrevExpander()
    exp.load_items([])
    monkeypatch.setattr(tool_mod, "get_abbrev_expander", lambda: exp)
    return exp


@pytest.mark.asyncio
async def test_execute_finds_program_by_user_phrasing(sessionmaker_bound, expander):
    """«у меня 250 баллов, есть шансы на ПИиКН» — баллы находятся (#235)."""
    await _seed_fit(sessionmaker_bound)

    for program in ("ПИиКН", "программная инженерия", "ПИиКН (Программная инженерия)"):
        text = await execute_admission_scores({"program": program})
        assert "246" in text, program
        assert "260" not in text, program


@pytest.mark.asyncio
async def test_execute_uses_admin_abbreviations(sessionmaker_bound, expander):
    """Сокращение, которое не выводится из названия, берётся из словаря админки."""
    await _seed_fit(sessionmaker_bound)
    expander.load_items(
        [{"short": "Прогинж", "full": "Программная инженерия и компьютерные науки"}]
    )

    text = await execute_admission_scores({"faculty": "ФИТ", "program": "Прогинж"})
    assert "246" in text
    assert "260" not in text


@pytest.mark.asyncio
async def test_execute_falls_back_to_published_years(sessionmaker_bound, expander):
    """За текущий год итогов ещё нет — показываем прошлые, а не «данных нет»."""
    await _seed_fit(sessionmaker_bound)

    text = await execute_admission_scores({"program": "ПИиКН", "year": 2026})
    assert "За 2026 год данных нет" in text
    assert "246" in text


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


@pytest.mark.asyncio
async def test_custom_scores_use_own_university(sessionmaker_bound, expander):
    from config.instance import install

    install({"UNIVERSITY_PROFILE": "custom", "UNIVERSITY_SHORT_NAME": "Север"}, 1)
    async with sessionmaker_bound() as session:
        await FacultyService(session).upsert_from_seed(
            [
                {
                    "name": "Факультет Севера",
                    "programs": [{"name": "Северная инженерия", "level": "bachelor"}],
                }
            ]
        )
        await AdmissionScoreService(session).upsert_from_rows(
            [
                ScoreRow(
                    faculty_name="Факультет Севера",
                    program_name="Северная инженерия",
                    code=None,
                    year=2025,
                    form="budget",
                    passing_score=210,
                    average_score=None,
                )
            ]
        )
    text = await default_tool_executor(
        "get_admission_scores", {"program": "Северная инженерия"}
    )
    assert "НГУ" not in text
    assert "Северная инженерия — Факультет Севера" in text
    assert "2025, бюджет: проходной 210" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["spo", "bachelor"])
async def test_program_directory_tool_separates_levels(
    sessionmaker_bound, monkeypatch, level
):
    from pipeline.tools import education_programs

    monkeypatch.setattr(education_programs, "AsyncSessionLocal", sessionmaker_bound)
    await _seed_fit(sessionmaker_bound)
    async with sessionmaker_bound() as session:
        service = FacultyService(session)
        faculty = await service.find_faculty_by_alias("ФИТ")
        assert faculty is not None
        await service.create_program(
            faculty.id,
            "Информационные системы и программирование",
            level="spo",
            code="09.02.07",
        )
    text = await default_tool_executor("get_education_programs", {"level": level})
    assert ("09.02.07" in text) == (level == "spo")
    assert ("09.03.04" in text) == (level == "bachelor")
    assert ("СПО" in text) == (level == "spo")


@pytest.mark.asyncio
async def test_scores_tool_retains_level_in_year_fallback(sessionmaker_bound, expander):
    await _seed_fit(sessionmaker_bound)
    text = await execute_admission_scores(
        {"faculty": "ФИТ", "level": "spo", "year": 2026}
    )
    assert text == "По этому запросу данных о проходных баллах нет."


def test_format_scores_distinguishes_same_names_at_different_levels():
    from pipeline.tools.admission_scores import _format_scores

    rows = [
        {"program_name": "Информатика", "level": "bachelor", "year": 2024},
        {"program_name": "Информатика", "level": "spo", "year": 2024},
    ]
    text = _format_scores(rows, "passing")
    assert "бакалавриат" in text
    assert "СПО" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["spo", "bachelor"])
async def test_scores_tool_separates_same_name_programs(
    sessionmaker_bound, expander, level
):
    await _seed_fit(sessionmaker_bound)
    name = "Программная инженерия и компьютерные науки"
    async with sessionmaker_bound() as session:
        service = FacultyService(session)
        faculty = await service.find_faculty_by_alias("ФИТ")
        assert faculty is not None
        await service.create_program(faculty.id, name, level="spo", code="09.02.07")
        await AdmissionScoreService(session).upsert_from_rows(
            [
                ScoreRow(
                    faculty_name="ФИТ",
                    program_name=name,
                    code="09.02.07",
                    level="spo",
                    year=2024,
                    form="budget",
                    passing_score=5,
                    average_score=None,
                )
            ]
        )
    text = await execute_admission_scores({"program": name, "level": level, "year": 2026})
    assert "За 2026 год данных нет" in text
    assert ("проходной 5" in text) == (level == "spo")
    assert ("246" in text) == (level == "bachelor")
    assert ("СПО" in text) == (level == "spo")


@pytest.mark.asyncio
async def test_unknown_level_does_not_broaden_tool_search():
    text = await default_tool_executor("get_education_programs", {"level": "unknown"})
    assert text == "Неизвестный уровень образования."
    text = await execute_admission_scores({"level": "unknown"})
    assert text == "Неизвестный уровень образования."


@pytest.mark.asyncio
async def test_program_tool_excludes_inactive_entries(sessionmaker_bound, monkeypatch):
    from pipeline.tools import education_programs

    monkeypatch.setattr(education_programs, "AsyncSessionLocal", sessionmaker_bound)
    async with sessionmaker_bound() as session:
        service = FacultyService(session)
        active = await service.create_faculty("Колледж")
        inactive = await service.create_faculty("Закрытое подразделение", is_active=False)
        await service.create_program(active.id, "Доступная программа", level="spo")
        await service.create_program(
            active.id, "Архивная программа", level="spo", is_active=False
        )
        await service.create_program(inactive.id, "Скрытая программа", level="spo")
    text = await default_tool_executor("get_education_programs", {"level": "spo"})
    assert "Доступная программа" in text
    assert "Архивная программа" not in text
    assert "Скрытая программа" not in text
    for faculty in ("Колледж", "Закрытое подразделение"):
        text = await default_tool_executor(
            "get_education_programs", {"level": "spo", "faculty": faculty}
        )
        assert "Архивная программа" not in text
        assert "Скрытая программа" not in text
        assert ("Доступная программа" in text) == (faculty == "Колледж")
