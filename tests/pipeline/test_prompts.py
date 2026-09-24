"""Промпты вынесены из логики (#313) — и хрупкие места стали менее заметны.

Код ответа полагается на текст промпта: _postprocess отключает источники по
формуле отказа, _run_tool_loop всегда предлагает get_admission_scores, а
сборка промпта — на плейсхолдеры. Эти инварианты фиксируются здесь, чтобы
правка «просто строки» не прошла молча. Версии нужны евалам: по ним прогон
связывается с конкретным текстом промпта.
"""

import re

import pytest

from pipeline.prompts import (
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
    ("system", SYSTEM_PROMPT_VERSION, SYSTEM_PROMPT_BASE),
    ("sources_hint", SOURCES_HINT_VERSION, SOURCES_HINT),
    ("lightrag_level", LIGHTRAG_LEVEL_HINT_VERSION, LIGHTRAG_LEVEL_HINT),
    ("lightrag_format", LIGHTRAG_FORMAT_HINT_VERSION, LIGHTRAG_FORMAT_HINT),
    ("intent", INTENT_PROMPT_VERSION, INTENT_PROMPT_TEMPLATE),
]


class TestVersions:
    @pytest.mark.parametrize(
        "name, version, text",
        VERSIONED_PROMPTS,
        ids=[row[0] for row in VERSIONED_PROMPTS],
    )
    def test_every_prompt_has_version(self, name, version, text):
        """Изменение текста промпта без bump версии должно падать здесь."""
        assert re.fullmatch(r"\d+\.\d+", version), f"{name}: версия не X.Y"
        assert text.strip(), f"{name}: пустой текст"


class TestSystemPrompt:
    def test_placeholders_are_substituted(self):
        prompt = SYSTEM_PROMPT_BASE.format(
            context="Приём до 25 июля.", sources_hint=SOURCES_HINT
        )
        assert "Приём до 25 июля." in prompt
        assert "ИНСТРУКЦИЯ К ОТВЕТУ" in prompt
        # Незаполненный плейсхолдер ушёл бы модели как мусорные скобки.
        assert "{" not in prompt
        assert "}" not in prompt

    def test_refusal_formula_is_word_for_word(self):
        """_postprocess ищет эту фразу и по ней отключает источники:
        изменить её в промпте можно только вместе с постобработкой."""
        assert "Я не нашел информации об этом в базе знаний НГУ" in (SYSTEM_PROMPT_BASE)

    def test_admission_scores_tool_is_named(self):
        """Правило про проходные баллы опирается на инструмент, который
        _run_tool_loop предлагает модели всегда."""
        assert "get_admission_scores" in SYSTEM_PROMPT_BASE


class TestIntentPrompt:
    def test_topics_placeholder_formats(self):
        """Скобки JSON-примера удвоены под str.format: сломанное экранирование
        уронит фоновую классификацию целиком (и молча — там try/except)."""
        prompt = INTENT_PROMPT_TEMPLATE.format(topics_list="1: Приём")
        assert '{"is_nsu": true' in prompt
        assert "1: Приём" in prompt
        assert "{" not in prompt.replace('{"is_nsu"', "")
