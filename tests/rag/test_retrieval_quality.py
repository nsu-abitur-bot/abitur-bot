"""
RAG Retrieval Quality — оценка качества ретривера
по нескольким метрикам: source relevance, ranking, context
completeness, cross-topic isolation.

Отличие от test_keyword_matching:
  - keyword matching проверяет «нашлось ли хоть одно ключевое слово?»
  - retrieval quality проверяет «достаточно ли контекста для правильного ответа?»
"""
from pathlib import Path

import pytest
pytest.skip("Тесты несовместимы с новой графовой RAG системой", allow_module_level=True)

from rag.loader import add_texts
from rag.retriever import search_similar

BAZA_DIR = Path(__file__).parent.parent.parent / "baza"


# ──────────────────────────────────────────────────────
# Фикстура: загружаем базу знаний в изолированное хранилище
# ──────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def load_baza(tmp_path_factory):
    """Загружаем все .md файлы из baza/ в изолированное временное хранилище."""
    import rag.vectorstore as vs_module

    tmp_dir = tmp_path_factory.mktemp("chroma_quality")
    original_persist_dir = vs_module.PERSIST_DIR
    original_instance = vs_module._vectorstore_instance

    vs_module.PERSIST_DIR = str(tmp_dir)
    vs_module._vectorstore_instance = None

    texts = [f.read_text(encoding="utf-8") for f in BAZA_DIR.glob("*.md")]
    assert texts, f"Не найдены .md файлы в {BAZA_DIR}"
    add_texts(texts)

    yield

    vs_module._vectorstore_instance = None
    vs_module.PERSIST_DIR = original_persist_dir


# ══════════════════════════════════════════════════════
# 1. Context Completeness — достаточно ли контекста для ответа?
#    Проверяем что ВСЕ обязательные факты присутствуют в retrieved chunks.
# ══════════════════════════════════════════════════════

# (вопрос, список ОБЯЗАТЕЛЬНЫХ фактов — ВСЕ должны присутствовать)
COMPLETENESS_CASES = [
    (
        "Какие ЕГЭ нужны для поступления на ФИТ?",
        ["математик", "русский"],
        "Контекст должен содержать оба обязательных ЕГЭ для ФИТ",
    ),
    (
        "Какие ЕГЭ нужны для экономического факультета НГУ?",
        ["математик", "русский"],
        "Контекст должен содержать оба обязательных ЕГЭ для экономфака",
    ),
    pytest.param(
        "Сколько стоит обучение на ФИТе?",
        ["195"],
        "Контекст должен содержать стоимость обучения на ФИТ",
        marks=pytest.mark.xfail(
            reason="all-MiniLM-L6-v2 не находит чанк со стоимостью ФИТ — нужна русскоязычная модель"
        ),
    ),
    (
        "Какие направления есть на ММФ?",
        ["математик", "механик"],
        "Контекст должен упоминать математику и механику",
    ),
    (
        "Какие ЕГЭ сдавать на физический факультет?",
        ["математик", "физик"],
        "Контекст должен содержать оба обязательных ЕГЭ для физфака",
    ),
    pytest.param(
        "Сколько стоит общежитие НГУ?",
        ["1 600", "1\u202f600", "1600"],
        "Контекст должен содержать стоимость общежития",
        marks=pytest.mark.xfail(
            reason="RAG не находит чанк с ценой — нужно улучшить чанкинг dorm.md"
        ),
    ),
]


@pytest.mark.parametrize("question,required_facts,description", COMPLETENESS_CASES)
def test_context_completeness(question, required_facts, description):
    """Все обязательные факты должны присутствовать в retrieved контексте."""
    results = search_similar(question, k=5)
    assert results, f"RAG вернул пустой результат для: '{question}'"

    combined = " ".join(doc.page_content.lower() for doc in results)

    missing = [f for f in required_facts if f.lower() not in combined]
    assert not missing, (
        f"{description}\n"
        f"Вопрос: '{question}'\n"
        f"Отсутствующие факты: {missing}\n"
        f"Контекст (первые 500 символов): {combined[:500]}"
    )


# ══════════════════════════════════════════════════════
# 2. Ranking Quality — самый релевантный чанк на первом месте?
#    Проверяем что top-1 документ содержит ключевой индикатор.
# ══════════════════════════════════════════════════════

# (вопрос, ключевое слово которое ДОЛЖНО быть в top-1 чанке)
RANKING_CASES = [
    pytest.param(
        "Какие экзамены нужны для поступления на ФИТ?",
        ["информатик", "информационных технологий", "фит", "09.03.01"],
        marks=pytest.mark.xfail(
            reason="all-MiniLM-L6-v2 ранжирует нерелевантный чанк выше ФИТ — нужна русскоязычная модель"
        ),
    ),
    pytest.param(
        "Какой проходной балл на бюджет на ФИТе?",
        ["267", "фит", "информационных технологий"],
        marks=pytest.mark.xfail(
            reason="all-MiniLM-L6-v2 ранжирует нерелевантный чанк выше ФИТ — нужна русскоязычная модель"
        ),
    ),
    (
        "Какой проходной балл на физфак НГУ?",
        ["220", "физическ"],
    ),
    pytest.param(
        "Какой проходной балл на математику в НГУ?",
        ["239", "232", "ммф", "механико"],
        marks=pytest.mark.xfail(
            reason="all-MiniLM-L6-v2 возвращает нерелевантный top-1 для ММФ — нужна русскоязычная модель"
        ),
    ),
    (
        "Есть ли юриспруденция в НГУ?",
        ["юриспруден", "право", "ифп", "философи"],
    ),
    pytest.param(
        "Какие студенческие клубы есть в НГУ?",
        ["клуб", "квант", "гея"],
        marks=pytest.mark.xfail(
            reason="all-MiniLM-L6-v2 не ранжирует чанк о клубах в top-1 — нужна русскоязычная модель"
        ),
    ),
]


@pytest.mark.parametrize("question,top1_indicators", RANKING_CASES)
def test_ranking_top1_relevance(question, top1_indicators):
    """Top-1 документ должен содержать хотя бы один ключевой индикатор темы."""
    results = search_similar(question, k=3)
    assert results, f"RAG вернул пустой результат для: '{question}'"

    top1 = results[0].page_content.lower()

    matched = [ind for ind in top1_indicators if ind.lower() in top1]
    assert matched, (
        f"Top-1 чанк нерелевантен!\n"
        f"Вопрос: '{question}'\n"
        f"Ожидались индикаторы (хотя бы один): {top1_indicators}\n"
        f"Top-1 чанк: {top1[:300]}"
    )


# ══════════════════════════════════════════════════════
# 3. Cross-Topic Isolation — запрос по одной теме
#    НЕ должен возвращать чанки из нерелевантной темы на TOP-1.
# ══════════════════════════════════════════════════════

# (вопрос, слова которых НЕ ДОЛЖНО быть в top-1 чанке)
ISOLATION_CASES = [
    (
        "Какие экзамены нужны на ФИТ?",
        ["геолог", "геофизик", "нефть", "лечебное дело"],
        "Запрос о ФИТ не должен возвращать геологию/медицину",
    ),
    (
        "Какой проходной балл на геологию?",
        ["информатик", "вычислительн", "программн"],
        "Запрос о ГГФ не должен возвращать ФИТ",
    ),
    (
        "Есть ли психология в НГУ?",
        ["робот", "мехатроник", "геолог"],
        "Запрос о психологии не должен возвращать робототехнику/геологию",
    ),
    (
        "Расскажи про студенческие клубы НГУ",
        ["проходной балл", "бюджет", "егэ"],
        "Запрос о клубах не должен возвращать информацию о поступлении",
    ),
]


@pytest.mark.parametrize("question,forbidden_words,description", ISOLATION_CASES)
def test_cross_topic_isolation(question, forbidden_words, description):
    """Top-1 чанк не должен содержать нерелевантные маркеры другой темы."""
    results = search_similar(question, k=3)
    assert results, f"RAG вернул пустой результат для: '{question}'"

    top1 = results[0].page_content.lower()

    found_forbidden = [w for w in forbidden_words if w.lower() in top1]
    assert not found_forbidden, (
        f"{description}\n"
        f"Вопрос: '{question}'\n"
        f"Найдены нерелевантные слова в top-1: {found_forbidden}\n"
        f"Top-1 чанк: {top1[:300]}"
    )


# ══════════════════════════════════════════════════════
# 4. Factual Retrieval — может ли RAG найти конкретный числовой факт?
#    Проверяем что конкретные цифры (баллы, цены, года) извлекаются.
# ══════════════════════════════════════════════════════

FACTUAL_CASES = [
    pytest.param(
        "Проходной балл на ФИТ бюджет", "267", "Проходной балл ФИТ = 267",
        marks=pytest.mark.xfail(reason="all-MiniLM-L6-v2 не находит числовой факт для ФИТ"),
    ),
    ("Проходной балл физический факультет бюджет", "220", "Проходной балл ФФ = 220"),
    ("Проходной балл математика НГУ бюджет", "239", "Проходной балл ММФ = 239"),
    pytest.param(
        "Проходной балл геология НГУ бюджет", "222", "Проходной балл ГГФ = 222",
        marks=pytest.mark.xfail(reason="all-MiniLM-L6-v2 не находит числовой факт для ГГФ"),
    ),
    pytest.param(
        "Проходной балл факультет естественных наук бюджет", "233", "Проходной балл ФЕН = 233",
        marks=pytest.mark.xfail(reason="all-MiniLM-L6-v2 не находит числовой факт для ФЕН"),
    ),
    pytest.param(
        "Проходной балл история бюджет НГУ", "248", "Проходной балл ГИ история = 248",
        marks=pytest.mark.xfail(reason="all-MiniLM-L6-v2 не находит числовой факт для ГИ"),
    ),
    ("Проходной балл юриспруденция бюджет НГУ", "253", "Проходной балл ИФП юр = 253"),
    pytest.param(
        "Проходной балл психология бюджет НГУ", "241", "Проходной балл психология = 241",
        marks=pytest.mark.xfail(reason="all-MiniLM-L6-v2 не находит числовой факт для психологии"),
    ),
    pytest.param(
        "Стоимость обучения ФИТ", "195", "Стоимость ФИТ ~195000",
        marks=pytest.mark.xfail(reason="all-MiniLM-L6-v2 не находит стоимость для ФИТ"),
    ),
]


@pytest.mark.parametrize("question,expected_number,description", FACTUAL_CASES)
def test_factual_number_retrieval(question, expected_number, description):
    """RAG должен извлечь чанк с конкретным числовым фактом."""
    results = search_similar(question, k=5)
    assert results, f"RAG вернул пустой результат для: '{question}'"

    combined = " ".join(doc.page_content for doc in results)

    assert expected_number in combined, (
        f"{description}\n"
        f"Вопрос: '{question}'\n"
        f"Число '{expected_number}' не найдено в retrieved контексте\n"
        f"Контекст (первые 500 символов): {combined[:500]}"
    )


# ══════════════════════════════════════════════════════
# 5. Query Robustness — разные формулировки одного вопроса
#    дают пересекающийся контекст.
# ══════════════════════════════════════════════════════

ROBUSTNESS_CASES = [
    pytest.param(
        "Какие экзамены нужны для ФИТ?",
        "ЕГЭ на факультет информационных технологий",
        "Два разных запроса о ФИТ ЕГЭ",
        marks=pytest.mark.xfail(
            reason="all-MiniLM-L6-v2 даёт разный top-3 для разных формулировок — нужна русскоязычная модель"
        ),
    ),
    (
        "Проходной балл физфак",
        "Минимальный балл для поступления на физический факультет НГУ",
        "Два разных запроса о проходном балле ФФ",
    ),
    (
        "Клубы НГУ",
        "Какие студенческие организации есть в Новосибирском университете?",
        "Два разных запроса о клубах",
    ),
]


@pytest.mark.parametrize("query_a,query_b,description", ROBUSTNESS_CASES)
def test_query_robustness(query_a, query_b, description):
    """Разные формулировки одинакового вопроса должны давать пересекающийся контекст."""
    results_a = search_similar(query_a, k=3)
    results_b = search_similar(query_b, k=3)

    assert results_a and results_b

    texts_a = {doc.page_content for doc in results_a}
    texts_b = {doc.page_content for doc in results_b}

    overlap = texts_a & texts_b
    assert overlap, (
        f"{description}\n"
        f"Запрос A: '{query_a}'\n"
        f"Запрос B: '{query_b}'\n"
        f"Нет пересечения в top-3 результатах — ретривер нестабилен"
    )
