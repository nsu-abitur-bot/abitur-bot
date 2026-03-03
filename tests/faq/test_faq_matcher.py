"""Тесты для модуля FAQ Matcher."""

from pathlib import Path

import pytest
import yaml

from faq.faq_matcher import FAQMatcher


# ── Фикстура: временный YAML-файл с FAQ ──────────────────────────────

SAMPLE_FAQ = {
    "faq": [
        {
            "question": "Какие факультеты есть в НГУ?",
            "aliases": [
                "Список факультетов НГУ",
                "Факультеты и институты НГУ",
            ],
            "answer": "В НГУ есть ФИТ, ММФ, ФФ, ФЕН и другие факультеты.",
        },
        {
            "question": "Сколько стоит общежитие в НГУ?",
            "aliases": ["Стоимость проживания в общежитии"],
            "answer": "От 1 600 рублей в месяц.",
        },
        {
            "question": "Какие ЕГЭ нужны на ФИТ?",
            "aliases": [],
            "answer": "Русский язык, математика, физика или информатика.",
        },
    ]
}


@pytest.fixture
def faq_yaml(tmp_path: Path) -> Path:
    """Создаёт временный YAML-файл с тестовыми FAQ."""
    faq_file = tmp_path / "test_faq.yaml"
    with open(faq_file, "w", encoding="utf-8") as f:
        yaml.dump(SAMPLE_FAQ, f, allow_unicode=True)
    return faq_file


@pytest.fixture
def matcher(faq_yaml: Path) -> FAQMatcher:
    """Создаёт FAQMatcher с тестовыми данными."""
    return FAQMatcher(faq_path=faq_yaml, threshold=0.80)


# ── Тесты ─────────────────────────────────────────────────────────────


class TestFAQMatcherLoading:
    """Тесты загрузки FAQ."""

    def test_loads_entries(self, matcher: FAQMatcher):
        # 3 вопроса + 3 alias'а = 6 фраз
        assert matcher.size == 6

    def test_empty_file(self, tmp_path: Path):
        faq_file = tmp_path / "empty.yaml"
        faq_file.write_text("faq: []", encoding="utf-8")
        m = FAQMatcher(faq_path=faq_file)
        assert m.size == 0

    def test_missing_file(self, tmp_path: Path):
        m = FAQMatcher(faq_path=tmp_path / "nonexistent.yaml")
        assert m.size == 0
        assert m.match("любой вопрос") is None


class TestFAQMatcherMatching:
    """Тесты семантического сопоставления."""

    def test_exact_match(self, matcher: FAQMatcher):
        result = matcher.match("Какие факультеты есть в НГУ?")
        assert result is not None
        assert "ФИТ" in result

    def test_alias_match(self, matcher: FAQMatcher):
        result = matcher.match("Список факультетов НГУ")
        assert result is not None
        assert "ФИТ" in result

    def test_paraphrased_match(self, matcher: FAQMatcher):
        result = matcher.match("Какие есть факультеты в Новосибирском университете?")
        assert result is not None
        assert "ФИТ" in result

    def test_dormitory_question(self, matcher: FAQMatcher):
        result = matcher.match("Сколько стоит жить в общежитии НГУ?")
        assert result is not None
        assert "1 600" in result

    def test_no_match_unrelated(self, matcher: FAQMatcher):
        result = matcher.match("Какая погода в Москве сегодня?")
        assert result is None

    def test_no_match_gibberish(self, matcher: FAQMatcher):
        result = matcher.match("asdfghjkl qwerty")
        assert result is None


class TestFAQMatcherThreshold:
    """Тесты порога сходства."""

    def test_high_threshold_rejects(self, faq_yaml: Path):
        m = FAQMatcher(faq_path=faq_yaml, threshold=0.99)
        # Даже перефразированный вопрос не пройдёт при пороге 0.99
        result = m.match("В НГУ какие факультеты бывают?")
        assert result is None

    def test_low_threshold_accepts_more(self, faq_yaml: Path):
        m = FAQMatcher(faq_path=faq_yaml, threshold=0.50)
        result = m.match("Что есть в НГУ?")
        assert result is not None

    def test_threshold_property(self, matcher: FAQMatcher):
        assert matcher.threshold == 0.80
        matcher.threshold = 0.90
        assert matcher.threshold == 0.90


class TestFAQMatcherReload:
    """Тесты перезагрузки FAQ."""

    def test_reload_updates_data(self, faq_yaml: Path):
        m = FAQMatcher(faq_path=faq_yaml, threshold=0.80)
        assert m.size == 6

        # Добавляем новый FAQ
        with open(faq_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        data["faq"].append(
            {
                "question": "Новый вопрос",
                "aliases": [],
                "answer": "Новый ответ",
            }
        )

        with open(faq_yaml, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        m.reload()
        assert m.size == 7
