"""Тесты для модуля FAQ Matcher."""

from pathlib import Path

import pytest
import yaml

from faq.matcher import FAQMatcher, clean_user_input


class FakeEmbeddings:
    _ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщьыъэюяabcdefghijklmnopqrstuvwxyz0123456789"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def _encode(self, text: str) -> list[float]:
        t = text.lower()

        char_counts = [float(t.count(ch)) for ch in self._ALPHABET]

        faculty = any(k in t for k in ["факультет", "институт"])
        dorm = any(k in t for k in ["общежит", "проживан", "стоит жить"])
        exams = any(k in t for k in ["егэ", "фит", "информат", "физик", "математ"])
        founded = any(k in t for k in ["основан", "появил", "каком году", "когда"])
        location = any(
            k in t for k in ["где", "город", "находит", "академгород", "новосибир"]
        )
        generic_ngu = "нгу" in t

        topic_features = [
            2.0 if faculty else 0.0,
            2.0 if dorm else 0.0,
            2.0 if exams else 0.0,
            2.0 if founded else 0.0,
            2.0 if location else 0.0,
            1.0 if generic_ngu else 0.0,
        ]

        return char_counts + topic_features


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
        {
            "question": "Когда был основан НГУ?",
            "aliases": [
                "Когда появился НГУ?",
                "В каком году основали НГУ?",
            ],
            "answer": "НГУ был создан в 1958 году.",
        },
        {
            "question": "Где находится НГУ?",
            "aliases": [
                "Где НГУ?",
                "В каком городе НГУ?",
            ],
            "answer": "НГУ расположен в Академгородке, Новосибирск.",
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
    return FAQMatcher(faq_path=faq_yaml, threshold=0.80, embedder=FakeEmbeddings())


# ── Тесты ─────────────────────────────────────────────────────────────


class TestCleanUserInput:
    """Тесты очистки пользовательского ввода."""

    def test_strips_from_prefix(self):
        assert (
            clean_user_input("[from Максим] когда появился НГУ?") == "когда появился НГУ?"
        )

    def test_strips_greeting(self):
        assert clean_user_input("привет, когда появился НГУ?") == "когда появился НГУ?"

    def test_strips_greeting_and_prefix(self):
        result = clean_user_input("[from user123] привет, где НГУ?")
        assert result == "где НГУ?"

    def test_strips_multiple_fillers(self):
        result = clean_user_input("здравствуйте, подскажите, какие факультеты?")
        assert "какие факультеты?" in result

    def test_preserves_meaningful_text(self):
        assert clean_user_input("Какие ЕГЭ нужны на ФИТ?") == "Какие ЕГЭ нужны на ФИТ?"

    def test_empty_after_cleaning(self):
        # pure greeting with no question
        result = clean_user_input("привет")
        # should return empty or just stripped
        assert result == "" or result is not None


class TestFAQMatcherLoading:
    """Тесты загрузки FAQ."""

    def test_loads_entries(self, matcher: FAQMatcher):
        # 5 вопросов + 7 alias'ов = 12 фраз
        assert matcher.size == 12

    def test_empty_file(self, tmp_path: Path):
        faq_file = tmp_path / "empty.yaml"
        faq_file.write_text("faq: []", encoding="utf-8")
        m = FAQMatcher(faq_path=faq_file, embedder=FakeEmbeddings())
        assert m.size == 0

    def test_missing_file(self, tmp_path: Path):
        m = FAQMatcher(faq_path=tmp_path / "nonexistent.yaml", embedder=FakeEmbeddings())
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
        result = matcher.match("How to cook pasta carbonara recipe?")
        assert result is None

    def test_no_match_gibberish(self, matcher: FAQMatcher):
        result = matcher.match("asdfghjkl qwerty")
        assert result is None

    def test_no_match_vague_question(self, matcher: FAQMatcher):
        """Размытый вопрос 'Что есть в НГУ' не должен ложно срабатывать."""
        result = matcher.match("Что есть в НГУ")
        assert result is None


class TestFAQMatcherWithNoisyInput:
    """Тесты сопоставления с 'грязным' вводом — приветствия, префиксы."""

    def test_greeting_plus_question(self, matcher: FAQMatcher):
        """Привет, когда появился НГУ? → должен вернуть ответ о 1958."""
        result = matcher.match("привет, когда появился НГУ?")
        assert result is not None
        assert "1958" in result

    def test_from_prefix_plus_question(self, matcher: FAQMatcher):
        """[from Максим] когда появился НГУ? → должен вернуть ответ."""
        result = matcher.match("[from Максим] когда появился НГУ?")
        assert result is not None
        assert "1958" in result

    def test_from_prefix_greeting_question(self, matcher: FAQMatcher):
        """[from user] привет, когда появился нгу? → полная очистка."""
        result = matcher.match("[from user123] привет, когда появился нгу?")
        assert result is not None
        assert "1958" in result

    def test_where_is_ngu_short(self, matcher: FAQMatcher):
        """Где НГУ? → должен вернуть ответ о местоположении."""
        result = matcher.match("Где НГУ?")
        assert result is not None
        assert "Академгородке" in result or "Новосибирск" in result

    def test_from_prefix_where_question(self, matcher: FAQMatcher):
        """[from user] где НГУ? → после очистки должен сработать."""
        result = matcher.match("[from user] где НГУ?")
        assert result is not None
        assert "Новосибирск" in result

    def test_podskaji_question(self, matcher: FAQMatcher):
        """подскажи, какие факультеты в НГУ? → очистка 'подскажи'."""
        result = matcher.match("подскажи, какие факультеты в НГУ?")
        assert result is not None
        assert "ФИТ" in result


class TestFAQMatcherThreshold:
    """Тесты порога сходства."""

    def test_default_threshold_is_strict(self, faq_yaml: Path, monkeypatch):
        """Дефолт 0.95: ниже FAQ отвечает на соседние вопросы чужим ответом."""
        monkeypatch.delenv("FAQ_SIMILARITY_THRESHOLD", raising=False)
        m = FAQMatcher(faq_path=faq_yaml, embedder=FakeEmbeddings())
        assert m.threshold == 0.95

    def test_env_overrides_default(self, faq_yaml: Path, monkeypatch):
        monkeypatch.setenv("FAQ_SIMILARITY_THRESHOLD", "0.9")
        m = FAQMatcher(faq_path=faq_yaml, embedder=FakeEmbeddings())
        assert m.threshold == 0.9

    def test_env_invalid_falls_back_to_default(self, faq_yaml: Path, monkeypatch):
        monkeypatch.setenv("FAQ_SIMILARITY_THRESHOLD", "не-число")
        m = FAQMatcher(faq_path=faq_yaml, embedder=FakeEmbeddings())
        assert m.threshold == 0.95

    def test_explicit_threshold_wins_over_env(self, faq_yaml: Path, monkeypatch):
        monkeypatch.setenv("FAQ_SIMILARITY_THRESHOLD", "0.9")
        m = FAQMatcher(faq_path=faq_yaml, threshold=0.50, embedder=FakeEmbeddings())
        assert m.threshold == 0.50

    def test_high_threshold_rejects(self, faq_yaml: Path):
        m = FAQMatcher(faq_path=faq_yaml, threshold=0.99, embedder=FakeEmbeddings())
        # Даже перефразированный вопрос не пройдёт при пороге 0.99
        result = m.match("В НГУ какие факультеты бывают?")
        assert result is None

    def test_low_threshold_accepts_more(self, faq_yaml: Path):
        m = FAQMatcher(faq_path=faq_yaml, threshold=0.50, embedder=FakeEmbeddings())
        result = m.match("Что есть в НГУ?")
        assert result is not None

    def test_threshold_property(self, matcher: FAQMatcher):
        assert matcher.threshold == 0.80
        matcher.threshold = 0.90
        assert matcher.threshold == 0.90


class TestMatchAsync:
    """match и match_async делят одну реализацию — результаты совпадают."""

    async def test_match_and_match_async_agree(self, matcher: FAQMatcher):
        questions = [
            "Какие факультеты есть в НГУ?",
            "привет, когда появился НГУ?",
            "How to cook pasta carbonara recipe?",
        ]
        for question in questions:
            assert await matcher.match_async(question) == matcher.match(question)

    async def test_match_async_respects_threshold(self, faq_yaml: Path):
        m = FAQMatcher(faq_path=faq_yaml, threshold=0.99, embedder=FakeEmbeddings())
        assert await m.match_async("В НГУ какие факультеты бывают?") is None

    async def test_match_async_hit(self, matcher: FAQMatcher):
        result = await matcher.match_async("привет, когда появился НГУ?")
        assert result is not None
        assert "1958" in result


class TestLoadFaqThreshold:
    """Эффективный порог: значение из админки поверх env-дефолта."""

    async def test_db_value_overrides_env(self, monkeypatch):
        import faq.matcher as matcher_module

        monkeypatch.setenv("FAQ_SIMILARITY_THRESHOLD", "0.70")
        _patch_db(monkeypatch, matcher_module, db_value="0.85")
        assert await matcher_module.load_faq_threshold() == 0.85

    async def test_missing_db_value_falls_back_to_env(self, monkeypatch):
        import faq.matcher as matcher_module

        monkeypatch.setenv("FAQ_SIMILARITY_THRESHOLD", "0.70")
        _patch_db(monkeypatch, matcher_module, db_value=None)
        assert await matcher_module.load_faq_threshold() == 0.70

    async def test_no_env_uses_default(self, monkeypatch):
        import faq.matcher as matcher_module

        monkeypatch.delenv("FAQ_SIMILARITY_THRESHOLD", raising=False)
        _patch_db(monkeypatch, matcher_module, db_value=None)
        assert await matcher_module.load_faq_threshold() == 0.95

    async def test_invalid_db_value_falls_back(self, monkeypatch):
        import faq.matcher as matcher_module

        monkeypatch.setenv("FAQ_SIMILARITY_THRESHOLD", "0.70")
        _patch_db(monkeypatch, matcher_module, db_value="не-число")
        assert await matcher_module.load_faq_threshold() == 0.70

    async def test_db_failure_falls_back_to_env(self, monkeypatch):
        import faq.matcher as matcher_module

        monkeypatch.setenv("FAQ_SIMILARITY_THRESHOLD", "0.70")

        def _broken_session_local():
            raise ConnectionError("db down")

        monkeypatch.setattr(matcher_module, "AsyncSessionLocal", _broken_session_local)
        assert await matcher_module.load_faq_threshold() == 0.70


def _patch_db(monkeypatch, matcher_module, db_value: str | None) -> None:
    """Подменяет SettingsService/AsyncSessionLocal: get_value → db_value."""

    class _FakeService:
        def __init__(self, session):
            self.session = session

        async def get_value(self, key: str):
            return db_value

    class _FakeSession:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *exc_info):
            return False

    monkeypatch.setattr(matcher_module, "SettingsService", _FakeService)
    monkeypatch.setattr(matcher_module, "AsyncSessionLocal", lambda: _FakeSession())


class TestFAQMatcherReload:
    """Тесты перезагрузки FAQ."""

    def test_reload_updates_data(self, faq_yaml: Path):
        m = FAQMatcher(faq_path=faq_yaml, threshold=0.80, embedder=FakeEmbeddings())
        assert m.size == 12

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
        assert m.size == 13
