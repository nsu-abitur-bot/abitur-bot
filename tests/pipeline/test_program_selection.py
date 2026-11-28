"""Тесты маршрутизации и подбора направлений по ЕГЭ (#467).

Проверяем, что запросы на подбор направлений по баллам/предметам ЕГЭ
(включая полный запрос и follow-up) не классифицируются как запросы требований
к одной карточке в program_exams, не уходят в ложное уточнение профиля,
корректно определяют уровень образования и не смешивают СПО/магистратуру.
"""

from typing import cast

import pytest
from lightrag import LightRAG

from rag.crag import detect_level
from rag.program_exams import (
    ProgramCard,
    _select_cards,
    can_query_program_exams,
    is_program_exam_question,
    retrieve_program_exams,
)
from rag.trace import LightragTrace

FULL_QUERY = (
    "У меня 200 баллов ЕГЭ в сумме за русский язык, профильную математику "
    "и информатику. Хочу на бюджет после 11 классов. Перечисли все подходящие "
    "направления СибГИУ."
)

FOLLOW_UP_QUERY = "А ты скажи мне ВСЕ направления для моих баллов и ЕГЭ а я сам решу уже"


def test_program_selection_not_misrouted_to_program_exams():
    """Запросы подбора направлений не должны активировать program_exams."""
    assert not is_program_exam_question(FULL_QUERY)
    assert not can_query_program_exams(FULL_QUERY, history=None)

    assert not is_program_exam_question(FOLLOW_UP_QUERY)
    # Даже если в истории было уточнение профиля, повторное требование
    # подобрать все направления не должно уходить в program_exams:
    history = (
        f"user: {FULL_QUERY}\n"
        "assistant: Уточните профиль, чтобы я назвал требования к ЕГЭ. Варианты: ..."
    )
    assert not can_query_program_exams(FOLLOW_UP_QUERY, history=history)


def test_detect_level_for_after_11_classes():
    """Поступление после 11 классов по ЕГЭ относится к бакалавриату."""
    assert detect_level(FULL_QUERY) == "bachelor"


@pytest.mark.asyncio
async def test_retrieve_program_exams_returns_none_for_selection_queries():
    """retrieve_program_exams должен возвращать None для запросов подбора."""
    trace = LightragTrace()
    cards = [
        ProgramCard(
            doc_id="1",
            url=None,
            title="09.03.01 Информатика",
            code="09.03.01",
            name="Информатика и вычислительная техника",
            profile="Информатика",
            level="bachelor",
        ),
        ProgramCard(
            doc_id="2",
            url=None,
            title="09.02.07 Информационные системы",
            code="09.02.07",
            name="Информационные системы и программирование",
            profile="",
            level="spo",
        ),
        ProgramCard(
            doc_id="3",
            url=None,
            title="44.03.01 Русский язык",
            code="44.03.01",
            name="Педагогическое образование",
            profile="Русский язык",
            level="bachelor",
        ),
    ]

    result_full = await retrieve_program_exams(
        cast(LightRAG, None), FULL_QUERY, cards, trace, max_sources=3
    )
    assert result_full is None

    result_followup = await retrieve_program_exams(
        cast(LightRAG, None), FOLLOW_UP_QUERY, cards, trace, max_sources=3
    )
    assert result_followup is None


def test_select_cards_does_not_match_exam_subjects_as_programs():
    """Когда код не указан, предметы ЕГЭ не должны матчиться как профиль программы."""
    trace = LightragTrace()
    cards = [
        ProgramCard(
            doc_id="1",
            url=None,
            title="44.03.01 Русский язык",
            code="44.03.01",
            name="Русский язык",
            profile="",
            level="bachelor",
        ),
        ProgramCard(
            doc_id="2",
            url=None,
            title="01.03.01 Математика",
            code="01.03.01",
            name="Математика",
            profile="Информатика",
            level="bachelor",
        ),
    ]

    query_with_subjects = (
        "Какие экзамены сдавать при поступлении, если я сдаю русский язык и информатику?"
    )
    selected, recognized = _select_cards(query_with_subjects, cards, trace)
    assert not selected
    assert recognized is None


def _subject_named_cards() -> list[ProgramCard]:
    return [
        ProgramCard(
            doc_id="math",
            url="https://example.test/01.03.01",
            title="01.03.01 Математика",
            code="01.03.01",
            name="Математика",
            profile="",
            level="bachelor",
        ),
        ProgramCard(
            doc_id="phys",
            url="https://example.test/03.03.02",
            title="03.03.02 Физика",
            code="03.03.02",
            name="Физика",
            profile="",
            level="bachelor",
        ),
        ProgramCard(
            doc_id="chem",
            url="https://example.test/04.03.01",
            title="04.03.01 Химия",
            code="04.03.01",
            name="Химия",
            profile="",
            level="bachelor",
        ),
        ProgramCard(
            doc_id="ped-ru",
            url="https://example.test/44.03.01-ru",
            title="44.03.01 Педагогическое образование — Русский язык",
            code="44.03.01",
            name="Педагогическое образование",
            profile="Русский язык",
            level="bachelor",
        ),
        ProgramCard(
            doc_id="ped-info",
            url="https://example.test/44.03.01-info",
            title="44.03.01 Педагогическое образование — Информатика",
            code="44.03.01",
            name="Педагогическое образование",
            profile="Информатика",
            level="bachelor",
        ),
    ]


@pytest.mark.parametrize(
    ("question", "expected_doc_ids"),
    [
        ("Какие ЕГЭ нужны на направление „Математика“?", ["math"]),
        ("Какие ЕГЭ нужны на направление «Математика»?", ["math"]),
        ("Какие ЕГЭ нужны на направление подготовки Математика?", ["math"]),
        ("Какие ЕГЭ нужны на программу Физика?", ["phys"]),
        ("Какие ЕГЭ нужны на специальность «Химия»?", ["chem"]),
        ("Математика, бакалавриат — какие ЕГЭ нужны?", ["math"]),
        ("Физика: какие ЕГЭ нужны?", ["phys"]),
        ("Какие ЕГЭ нужны на «Физика»?", ["phys"]),
        ("Какие ЕГЭ нужны на профиль «Русский язык»?", ["ped-ru"]),
        (
            "Какие ЕГЭ нужны на направление «Педагогическое образование», "
            "профиль «Информатика»?",
            ["ped-info"],
        ),
        (
            "Какие ЕГЭ нужны на направление «Математика», если я сдаю русский язык "
            "и информатика?",
            ["math"],
        ),
        ("Какие ЕГЭ нужны на 01.03.01 «Математика»?", ["math"]),
    ],
)
def test_select_cards_keeps_explicit_program_or_profile_names(
    question: str, expected_doc_ids: list[str]
):
    """Явно названные программы/профили (в т.ч. предметные) не отбрасываются."""
    selected, recognized = _select_cards(
        question, _subject_named_cards(), LightragTrace()
    )
    assert [card.doc_id for card in selected] == expected_doc_ids
    assert recognized is not None


@pytest.mark.parametrize(
    "question",
    [
        "Какие ЕГЭ сдавать, если сдаю математика, физика и русский язык?",
        "Сдаю русский язык и информатика, какие экзамены ещё нужны?",
        "У меня есть ЕГЭ: русский язык и математика — какие экзамены нужны?",
        "Какие предметы ЕГЭ нужны, если выбрал физика и информатика?",
        "Русский язык и математика — какие ещё ЕГЭ нужны?",
        "Обязателен ли русский язык на ЕГЭ?",
        "Нужна ли математика или физика для поступления на ЕГЭ?",
    ],
)
def test_select_cards_rejects_exam_subject_lists_and_general_subject_questions(
    question: str,
):
    """Перечисление предметов и общие вопросы о предметах не матчатся."""
    selected, recognized = _select_cards(
        question, _subject_named_cards(), LightragTrace()
    )
    assert selected == []
    assert recognized is None


def test_select_cards_with_code_ignores_subject_list_profile_false_positive():
    """При указании кода «сдаю русский язык» не сужает профиль до «Русский язык»."""
    cards = [c for c in _subject_named_cards() if c.code == "44.03.01"]

    ambiguous, _ = _select_cards(
        "Какие ЕГЭ нужны на 44.03.01, если я сдаю русский язык?",
        cards,
        LightragTrace(),
    )
    assert [c.doc_id for c in ambiguous] == ["ped-ru", "ped-info"]

    explicit, _ = _select_cards(
        "Какие ЕГЭ нужны на 44.03.01, профиль «Русский язык»?",
        cards,
        LightragTrace(),
    )
    assert [c.doc_id for c in explicit] == ["ped-ru"]
