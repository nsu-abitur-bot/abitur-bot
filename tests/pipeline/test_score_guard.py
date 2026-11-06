"""Регрессия: неполный прошлый ответ не доказывает отсутствие баллов в SQL."""

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from llm.base import LLMResult
from pipeline import llm_client
from pipeline.score_guard import SCORES_UNAVAILABLE, ScoreGuard
from pipeline.tools.admission_scores import AdmissionScoresResult


@pytest.mark.asyncio
async def test_database_error_with_known_program_identity_does_not_retry(monkeypatch):
    calls = []

    class Provider:
        async def generate_with_tools(self, messages, **kwargs):
            calls.append(1)
            await kwargs["tool_executor"]("get_admission_scores", {"metric": "average"})
            return LLMResult("Данных нет.")

    async def execute(name, args):
        return AdmissionScoresResult("Ошибка SQL", status="error")

    monkeypatch.setattr(llm_client, "get_llm_provider", Provider)
    text, _, _ = await llm_client._run_tool_loop(
        [
            AIMessage(
                content="141",
                additional_kwargs={
                    "admission_scores": {
                        "program": "09.03.01",
                        "metric": "passing",
                        "_program_ids": ["profile-a"],
                    }
                },
            ),
            HumanMessage(content="А средний?"),
        ],
        "known-profile-error",
        tool_executor=execute,
    )
    assert text == SCORES_UNAVAILABLE
    assert calls == [1]


@pytest.mark.parametrize(
    "question",
    [
        "Какой средний балл на Анализ данных?",
        "А на 09.04.01?",
        "А на магистратуру?",
    ],
)
def test_program_change_clears_previous_row_identity(question):
    guard = ScoreGuard.from_messages(
        [
            AIMessage(
                content="141",
                additional_kwargs={
                    "admission_scores": {
                        "program": "09.03.01",
                        "level": "bachelor",
                        "metric": "passing",
                        "_program_ids": ["profile-a"],
                    }
                },
            ),
            HumanMessage(content=question),
        ]
    )
    assert guard is not None and not guard.program_ids
    if "магистратуру" not in question:
        assert "level" not in guard.filters


def test_identity_recovery_requires_all_previously_selected_profiles():
    from pipeline.tools import ADMISSION_SCORES_TOOL

    guard = ScoreGuard.from_messages(
        [
            AIMessage(
                content="141",
                additional_kwargs={
                    "admission_scores": {
                        "program": "09.03.01",
                        "metric": "passing",
                        "_program_ids": ["profile-a", "profile-b"],
                    }
                },
            ),
            HumanMessage(content="А средний?"),
        ]
    )
    assert guard is not None
    assert not guard.accepts({"metric": "average", "program_ids": ["profile-b"]})
    assert not guard.accepts({"metric": "average", "program_ids": ["profile-c"]})
    schema = guard.recovery_tool(ADMISSION_SCORES_TOOL).parameters
    assert schema["properties"]["program_ids"]["items"]["enum"] == [
        "profile-a",
        "profile-b",
    ]
    assert schema["properties"]["program_ids"]["minItems"] == 2
    assert "program_ids" in schema["required"]
    assert "program_ids" not in ADMISSION_SCORES_TOOL.parameters["properties"]


def test_broader_sql_result_does_not_expand_known_profile_or_requested_filters():
    guard = ScoreGuard(
        filters={
            "program": "09.03.01",
            "year": 2024,
            "form": "budget",
            "metric": "average",
        },
        program_ids=frozenset({"profile-a"}),
    )
    scope = guard.history_scope(
        {
            "program": "09.03.01",
            "year": 2023,
            "form": "paid",
            "metric": "both",
            "_program_ids": ["profile-a", "profile-b"],
        }
    )
    assert scope["_program_ids"] == ["profile-a"]
    assert (scope["year"], scope["form"], scope["metric"]) == (2024, "budget", "average")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "changes"),
    [
        ("Какой средний балл?", {"metric": "average"}),
        ("А на 09.03.04?", {"program": "09.03.04"}),
        ("А по платному?", {"form": "paid"}),
    ],
)
async def test_review_followups_require_matching_query_before_publication(
    monkeypatch, question, changes
):
    scope = {"program": "09.03.01", "year": 2024, "form": "budget", "metric": "passing"}
    expected = {**scope, **changes}
    calls, queries, published = [], [], []

    class Provider:
        async def generate_with_tools(self, messages, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                if question.startswith("Какой"):
                    await kwargs["tool_executor"](
                        "get_admission_scores",
                        {
                            "program": "38.03.01",
                            "year": 2025,
                            "form": "paid",
                            "metric": "average",
                        },
                    )
                if kwargs.get("on_delta"):
                    await kwargs["on_delta"]("Данных нет.")
                return LLMResult("Данных нет.")
            assert kwargs["required_tool"] == "get_admission_scores"
            properties = kwargs["tools"][0].parameters["properties"]
            for key, value in expected.items():
                assert properties[key]["enum"] == [value]
            await kwargs["tool_executor"]("get_admission_scores", expected)
            return LLMResult("Проверенный ответ")

    async def execute(name, args):
        queries.append(args)
        return AdmissionScoresResult("Баллы", status="ok", scope=dict(args))

    async def publish(text):
        published.append(text)

    monkeypatch.setattr(llm_client, "get_llm_provider", Provider)
    result, _, _ = await llm_client._run_tool_loop(
        [
            HumanMessage(content="Какой проходной балл на 09.03.01?"),
            AIMessage(
                content="Проходной 141", additional_kwargs={"admission_scores": scope}
            ),
            HumanMessage(content=question),
        ],
        "review-regressions",
        stream_callback=publish,
        tool_executor=execute,
    )
    assert result == "Проверенный ответ"
    assert queries == [expected]
    assert published == [result]
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("question", ["А средний балл?", "Какой средний балл?"])
async def test_generic_score_without_context_clarifies_without_llm(monkeypatch, question):
    def forbidden_provider():
        raise AssertionError("Уточнение направления не требует LLM")

    monkeypatch.setattr(llm_client, "get_llm_provider", forbidden_provider)
    result, _, _ = await llm_client._run_tool_loop(
        [HumanMessage(content=question)], "missing-score-context"
    )
    assert result == llm_client.SCORES_CLARIFY


def test_repeated_generic_questions_do_not_create_a_program_context():
    guard = ScoreGuard.from_messages(
        [
            HumanMessage(content="Какой средний балл?"),
            AIMessage(content=llm_client.SCORES_CLARIFY),
            HumanMessage(content="А средний балл?"),
        ]
    )
    assert guard is not None and guard.clarify


@pytest.mark.parametrize("question", ["А на 09.03.04?", "А по платному?"])
def test_score_shaped_followups_after_unrelated_topic_do_not_force_scores(question):
    assert (
        ScoreGuard.from_messages(
            [
                HumanMessage(content="Сколько стоит общежитие?"),
                AIMessage(content="1000 рублей"),
                HumanMessage(content=question),
            ]
        )
        is None
    )


@pytest.mark.asyncio
async def test_average_followup_does_not_publish_unchecked_absence(monkeypatch):
    published = []
    calls = []

    class Provider:
        async def generate_with_tools(self, messages, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                if kwargs.get("on_delta"):
                    await kwargs["on_delta"]("Среднего балла в базе нет.")
                return LLMResult("Среднего балла в базе нет.")
            await kwargs["tool_executor"](
                "get_admission_scores",
                {"program": "09.03.01", "metric": "average"},
            )
            return LLMResult("Средний балл — 63.27.")

    async def execute(name, args):
        from pipeline.tools.admission_scores import AdmissionScoresResult

        return AdmissionScoresResult("средний 63.27", status="ok")

    async def publish(text):
        published.append(text)

    monkeypatch.setattr(llm_client, "get_llm_provider", Provider)
    result, _, _ = await llm_client._run_tool_loop(
        [
            SystemMessage(content="Консультант по поступлению"),
            HumanMessage(content="Какой проходной балл на 09.03.01?"),
            AIMessage(content="2024, бюджет: проходной балл — 141."),
            HumanMessage(content="а средний бал какой?"),
        ],
        "score-guard-test",
        stream_callback=publish,
        tool_executor=execute,
    )
    assert result == "Средний балл — 63.27."
    assert published == [result]
    assert len(calls) == 2
    assert calls[1]["required_tool"] == "get_admission_scores"


@pytest.mark.parametrize(
    ("followups", "expected"),
    [
        (["а средний балл?"], {"program": "09.03.01", "metric": "average"}),
        (
            ["а средний?", "а в 2023 году?", "а на платку?"],
            {"program": "09.03.01", "metric": "average", "year": 2023, "form": "paid"},
        ),
        (
            ["а на магистратуру?"],
            {"metric": "passing", "level": "master"},
        ),
        (
            ["Какой проходной балл на 09.03.04?"],
            {"program": "09.03.04", "metric": "passing"},
        ),
    ],
)
def test_followup_scope(followups, expected):
    messages: list[BaseMessage] = [
        HumanMessage(content="Какой проходной балл на 09.03.01?")
    ]
    for question in followups:
        messages.extend([AIMessage(content="Ответ"), HumanMessage(content=question)])
    guard = ScoreGuard.from_messages(messages)
    assert guard is not None
    assert guard.filters == expected


def test_level_change_requires_new_program_instead_of_false_empty_query():
    guard = ScoreGuard.from_messages(
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
        ]
    )
    assert guard is not None and guard.clarify
    assert "program" not in guard.filters


def test_explicit_program_after_level_clarification_is_accepted():
    guard = ScoreGuard.from_messages(
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
        ]
    )
    assert guard is not None and not guard.clarify
    assert guard.accepts({"program": "09.04.01", "level": "master"})


@pytest.mark.parametrize(
    ("question", "key", "value"),
    [
        ("А средний балл за 2023 и 2024 годы?", "year", 2025),
        ("А средний балл за 2023 и 2024 годы?", "year", 2023),
        ("А средний балл на бюджет и платное?", "form", "budget"),
        ("А средний балл на бюджет и платное?", "form", "paid"),
    ],
)
def test_comparison_requires_complete_query(question, key, value):
    guard = ScoreGuard.from_messages(
        [
            AIMessage(
                content="141",
                additional_kwargs={
                    "admission_scores": {
                        "program": "09.03.01",
                        "year": 2024,
                        "form": "budget",
                        "metric": "passing",
                    }
                },
            ),
            HumanMessage(content=question),
        ]
    )
    assert guard is not None
    assert not guard.accepts({"program": "09.03.01", "metric": "average", key: value})
    assert guard.accepts({"program": "09.03.01", "metric": "average"})
    from pipeline.tools import ADMISSION_SCORES_TOOL

    schema = guard.recovery_tool(ADMISSION_SCORES_TOOL).parameters
    assert key not in schema["properties"]
    assert key not in schema["required"]


@pytest.mark.parametrize(
    "question",
    [
        "Какой средний балл за 2023 и 2024 годы?",
        "Какой средний балл на бюджет и платное?",
    ],
)
def test_full_comparison_question_retains_program_context(question):
    guard = ScoreGuard.from_messages(
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
            HumanMessage(content=question),
        ]
    )
    assert guard is not None
    assert guard.filters["program"] == "09.03.01"
    assert not guard.accepts({"program": "38.03.01", "metric": "average"})


@pytest.mark.parametrize(
    "question",
    [
        "Сколько стоит общежитие?",
        "Мой средний балл аттестата 4.5",
        "Как дела?",
    ],
)
def test_unrelated_questions_are_not_guarded(question):
    assert ScoreGuard.from_messages([HumanMessage(content=question)]) is None


def test_missing_scope_requires_clarification():
    guard = ScoreGuard.from_messages([HumanMessage(content="а средний?")])
    assert guard is not None and guard.clarify


def test_trusted_sql_scope_carries_faculty_year_and_form():
    guard = ScoreGuard.from_messages(
        [
            HumanMessage(content="Какой проходной балл на программную инженерию?"),
            AIMessage(
                content="Проходной 141",
                additional_kwargs={
                    "admission_scores": {
                        "program": "09.03.01",
                        "faculty": "Институт информационных технологий",
                        "level": "bachelor",
                        "year": 2024,
                        "form": "budget",
                        "metric": "passing",
                    }
                },
            ),
            HumanMessage(content="а средний балл?"),
        ]
    )
    assert guard is not None
    assert guard.filters == {
        "program": "09.03.01",
        "faculty": "Институт информационных технологий",
        "level": "bachelor",
        "year": 2024,
        "form": "budget",
        "metric": "average",
    }
    assert not guard.accepts({"faculty": "Другой институт", "metric": "average"})


def test_context_is_not_taken_from_unverified_answer_text():
    guard = ScoreGuard.from_messages(
        [
            HumanMessage(content="Какой проходной балл на 09.03.01?"),
            AIMessage(content="В 2025 году на платное — 123."),
            HumanMessage(content="а средний?"),
        ]
    )
    assert guard is not None
    assert "year" not in guard.filters and "form" not in guard.filters


def test_context_resets_after_unrelated_topic():
    assert (
        ScoreGuard.from_messages(
            [
                AIMessage(
                    content="Ответ",
                    additional_kwargs={
                        "admission_scores": {
                            "program": "09.03.01",
                            "metric": "passing",
                        }
                    },
                ),
                HumanMessage(content="Сколько стоит общежитие?"),
                AIMessage(content="1000 рублей"),
                HumanMessage(content="А в 2024?"),
            ]
        )
        is None
    )


def test_recovery_schema_preserves_known_scope_without_mutating_original():
    from pipeline.tools import ADMISSION_SCORES_TOOL

    guard = ScoreGuard(filters={"program": "09.03.01", "metric": "average", "year": 2024})
    tool = guard.recovery_tool(ADMISSION_SCORES_TOOL)
    assert tool.parameters["properties"]["year"]["enum"] == [2024]
    assert tool.parameters["properties"]["metric"]["enum"] == ["average"]
    assert set(tool.parameters["required"]) == {"program", "metric", "year"}
    assert "faculty" not in tool.parameters["properties"]
    assert "enum" not in ADMISSION_SCORES_TOOL.parameters["properties"]["year"]


def test_build_messages_keeps_trusted_scope_separate_from_prompt_text():
    messages = llm_client._build_messages(
        "",
        [
            {
                "role": "assistant",
                "content": "Проходной 141",
                "admission_scores": {"program": "09.03.01", "metric": "passing"},
            }
        ],
        "а средний?",
    )
    assert messages[1].additional_kwargs["admission_scores"]["program"] == "09.03.01"
    assert messages[1].content == "Проходной 141"


@pytest.mark.asyncio
async def test_valid_alias_query_does_not_need_recovery(monkeypatch):
    calls = []
    scope = {
        "program": "09.03.01",
        "faculty": "Факультет информационных технологий",
        "metric": "passing",
    }

    class Provider:
        async def generate_with_tools(self, messages, **kwargs):
            calls.append(kwargs)
            await kwargs["tool_executor"](
                "get_admission_scores",
                {"program": "КНиСТ", "faculty": "ФИТ", "metric": "average"},
            )
            return LLMResult("Средний 63.27")

    async def execute(name, args):
        return AdmissionScoresResult(
            "средний 63.27", status="ok", scope={**scope, "metric": "average"}
        )

    monkeypatch.setattr(llm_client, "get_llm_provider", Provider)
    text, _, _ = await llm_client._run_tool_loop(
        [
            AIMessage(
                content="Проходной 141", additional_kwargs={"admission_scores": scope}
            ),
            HumanMessage(content="А средний?"),
        ],
        "alias-query-test",
        tool_executor=execute,
    )
    assert text == "Средний 63.27"
    assert len(calls) == 1


def test_failed_followup_parameters_are_not_lost_before_next_question():
    guard = ScoreGuard.from_messages(
        [
            AIMessage(
                content="Проходной 141",
                additional_kwargs={
                    "admission_scores": {
                        "program": "09.03.01",
                        "year": 2024,
                        "metric": "passing",
                    }
                },
            ),
            HumanMessage(content="А в 2023?"),
            AIMessage(content="Не удалось проверить баллы."),
            HumanMessage(content="А на платку?"),
        ]
    )
    assert guard is not None
    assert guard.filters == {
        "program": "09.03.01",
        "year": 2023,
        "metric": "passing",
        "form": "paid",
    }


def test_new_named_program_does_not_force_previous_program_code():
    guard = ScoreGuard.from_messages(
        [
            AIMessage(
                content="Проходной 141",
                additional_kwargs={
                    "admission_scores": {
                        "program": "09.03.01",
                        "faculty": "ФИТ",
                        "year": 2024,
                        "metric": "passing",
                    }
                },
            ),
            HumanMessage(content="А средний балл на прикладную математику?"),
        ]
    )
    assert guard is not None
    assert guard.filters == {"year": 2024, "metric": "average"}


def test_free_education_does_not_mean_paid():
    guard = ScoreGuard.from_messages(
        [HumanMessage(content="Какой проходной балл на бесплатное обучение?")]
    )
    assert guard is not None and guard.filters["form"] == "budget"


@pytest.mark.asyncio
async def test_failed_validation_does_not_store_rejected_query_as_context(monkeypatch):
    from fakeredis.aioredis import FakeRedis

    history = object.__new__(llm_client.RedisClient)
    history.client = FakeRedis(decode_responses=True)
    ctx = llm_client._Ctx(session_id="rejected-scope", user_id=0)
    ctx.score_query = {"program": "09.03.04", "metric": "average"}
    tasks = []
    monkeypatch.setattr(llm_client, "_spawn_bg", tasks.append)
    llm_client._store_answer(SCORES_UNAVAILABLE, "А средний?", history, ctx)
    for task in tasks:
        await task
    entries = await history.get_history(ctx.session_id)
    await history.client.aclose()
    assert "admission_scores" not in entries[-1]


@pytest.mark.parametrize(
    "arguments",
    [
        {"program": "09.03.04", "metric": "average"},
        {"program": "09.03.01", "metric": "passing"},
        {"metric": "average", "year": 2025},
        {"metric": "average", "form": "paid"},
        {"metric": "average", "level": "master"},
    ],
)
def test_wrong_query_cannot_verify_answer(arguments):
    guard = ScoreGuard(
        filters={
            "program": "09.03.01",
            "metric": "average",
            "year": 2024,
            "form": "budget",
            "level": "bachelor",
        }
    )
    assert not guard.accepts(arguments)


def test_complete_broader_query_covers_subset():
    guard = ScoreGuard(filters={"metric": "average", "year": 2024, "form": "budget"})
    assert guard.accepts({"metric": "both"})


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["missing", "wrong", "error", "exception", "empty"])
async def test_recovery_is_bounded_and_error_is_not_empty(monkeypatch, mode):
    calls = []
    executed = []

    class Provider:
        async def generate_with_tools(self, messages, **kwargs):
            calls.append(kwargs)
            if mode != "missing":
                await kwargs["tool_executor"](
                    "get_admission_scores",
                    {"metric": "passing" if mode == "wrong" else "average"},
                )
            return LLMResult("Данных нет.")

    async def execute(name, args):
        executed.append(args)
        if mode == "exception":
            raise RuntimeError("database unavailable")
        return AdmissionScoresResult(
            "Данных нет.", status=("empty" if mode == "empty" else "error")
        )

    monkeypatch.setattr(llm_client, "get_llm_provider", Provider)
    result, _, _ = await llm_client._run_tool_loop(
        [HumanMessage(content="Средний балл на 09.03.01?")],
        "bounded-test",
        tool_executor=execute,
    )
    assert result == ("Данных нет." if mode == "empty" else SCORES_UNAVAILABLE)
    assert len(calls) == (2 if mode in ("missing", "wrong") else 1)
    if mode == "wrong":
        assert not executed
