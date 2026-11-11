"""Промпты вынесены из логики (#313) — и хрупкие места стали менее заметны.

Код ответа полагается на текст промпта: _postprocess отключает источники по
формуле отказа, _run_tool_loop всегда предлагает get_admission_scores, а
сборка промпта — на плейсхолдеры. Эти инварианты фиксируются здесь, чтобы
правка «просто строки» не прошла молча. Версии нужны евалам: по ним прогон
связывается с конкретным текстом промпта.
"""

import hashlib
import re

import pytest

from pipeline.prompts import (
    ANSWER_REVIEW_PROMPT,
    ANSWER_REVIEW_PROMPT_VERSION,
    INTENT_PROMPT_TEMPLATE,
    INTENT_PROMPT_VERSION,
    LIGHTRAG_FORMAT_HINT,
    LIGHTRAG_FORMAT_HINT_VERSION,
    LIGHTRAG_LEVEL_HINT,
    LIGHTRAG_LEVEL_HINT_VERSION,
    SOURCES_HINT,
    SOURCES_HINT_VERSION,
    SYSTEM_PROMPT_BASE,
    SYSTEM_PROMPT_VERSION,
)

VERSIONED_PROMPTS = [
    ("answer_review", ANSWER_REVIEW_PROMPT_VERSION, ANSWER_REVIEW_PROMPT),
    ("system", SYSTEM_PROMPT_VERSION, SYSTEM_PROMPT_BASE),
    ("sources_hint", SOURCES_HINT_VERSION, SOURCES_HINT),
    ("lightrag_level", LIGHTRAG_LEVEL_HINT_VERSION, LIGHTRAG_LEVEL_HINT),
    ("lightrag_format", LIGHTRAG_FORMAT_HINT_VERSION, LIGHTRAG_FORMAT_HINT),
    ("intent", INTENT_PROMPT_VERSION, INTENT_PROMPT_TEMPLATE),
]


# Версия и sha256 текста, с которыми промпт зафиксирован. Поменял текст —
# подними версию в pipeline/prompts.py и обнови обе колонки здесь.
PINNED = {
    "answer_review": (
        "3.0",
        "4e32bb54f09ccda531a4efc9f2d0704317ef39824a41ba271488cda1f57329b9",
    ),
    "system": (
        "4.0",
        "5b50de564ddab07f678db41dbade8d08b0173572ebbf6b40c49ba78f747ce9f7",
    ),
    "sources_hint": (
        "1.0",
        "781579fd770ab1e7da1c834a9b18f58c77d5042045b2841fe7ab8696edcff51a",
    ),
    "lightrag_level": (
        "1.0",
        "2a042ef6ebbe6d5c1e9797c210a8a9196f94ecdc210ef562dc516a34fc22dd80",
    ),
    "lightrag_format": (
        "1.0",
        "3d0831526a140ba86a58b86e1547b0a3374870fb7f40dae7370c77e22fee1de4",
    ),
    "intent": (
        "1.0",
        "109c9d4ef7fecd387dc37363d797d040cfba02c1585dab0c0ff8c3630fb7378e",
    ),
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class TestVersions:
    def test_every_prompt_is_pinned(self):
        assert {row[0] for row in VERSIONED_PROMPTS} == set(PINNED)

    @pytest.mark.parametrize(
        "name, version, text",
        VERSIONED_PROMPTS,
        ids=[row[0] for row in VERSIONED_PROMPTS],
    )
    def test_text_change_requires_version_bump(self, name, version, text):
        """Изменение текста промпта без bump версии падает здесь."""
        assert re.fullmatch(r"\d+\.\d+", version), f"{name}: версия не X.Y"
        pinned_version, pinned_sha = PINNED[name]
        if version == pinned_version:
            assert _sha256(text) == pinned_sha, (
                f"{name}: текст изменился, а версия осталась {version}. "
                "Подними версию и обнови PINNED."
            )
        else:
            pytest.fail(
                f"{name}: версия {pinned_version} → {version}. Обнови PINNED: "
                f'("{version}", "{_sha256(text)}")'
            )


class TestSystemPrompt:
    def test_placeholders_are_substituted(self):
        prompt = SYSTEM_PROMPT_BASE.format(
            context="Приём до 25 июля.",
            sources_hint=SOURCES_HINT,
            assistant_name="официальный дружелюбный помощник-бот",
            university_short_name="НГУ",
            university_full_name="Новосибирский государственный университет",
        )
        assert "Приём до 25 июля." in prompt
        assert "ИНСТРУКЦИЯ К ОТВЕТУ" in prompt
        # Незаполненный плейсхолдер ушёл бы модели как мусорные скобки.
        assert "{" not in prompt
        assert "}" not in prompt

    def test_refusal_formula_is_word_for_word(self):
        """_postprocess ищет эту фразу и по ней отключает источники:
        изменить её в промпте можно только вместе с постобработкой."""
        assert (
            "Я не нашел информации об этом в базе знаний НГУ"
            in SYSTEM_PROMPT_BASE.format(
                context="",
                sources_hint="",
                assistant_name="помощник",
                university_short_name="НГУ",
                university_full_name="НГУ",
            )
        )

    def test_admission_scores_tool_is_named(self):
        """Правило про проходные баллы опирается на инструмент, который
        _run_tool_loop предлагает модели всегда."""
        assert "get_admission_scores" in SYSTEM_PROMPT_BASE.format(
            context="",
            sources_hint="",
            assistant_name="помощник",
            university_short_name="НГУ",
            university_full_name="НГУ",
        )


class TestIntentPrompt:
    def test_topics_placeholder_formats(self):
        """Скобки JSON-примера удвоены под str.format: сломанное экранирование
        уронит фоновую классификацию целиком (и молча — там try/except)."""
        prompt = INTENT_PROMPT_TEMPLATE.format(topics_list="1: Приём")
        assert '{"is_nsu": true' in prompt
        assert "1: Приём" in prompt
        assert "{" not in prompt.replace('{"is_nsu"', "")


def test_empty_program_catalog_does_not_override_knowledge_base() -> None:
    assert (
        "Пустой результат get_education_programs относится только к справочнику"
        in SYSTEM_PROMPT_BASE
    )
