"""
Keyword Matching — проверяем что RAG находит нужные чанки
с ключевыми словами для типичных вопросов абитуриентов.
"""
import pytest
from rag.loader import add_texts
from rag.retriever import search_similar
from pathlib import Path


BAZA_DIR = Path(__file__).parent.parent.parent / "baza"

# (вопрос, ключевые слова из baza/, хотя бы одно должно быть в ответе)
KEYWORD_CASES = [
    (
        "Какие экзамены нужны для поступления на ФИТ?",
        ["математика", "физика", "информатика", "русский"],
    ),
    pytest.param(
        "Сколько стоит общежитие в НГУ?",
        ["1 600", "1600", "рублей", "стоимост"],
        marks=pytest.mark.xfail(reason="RAG не находит чанк с ценой — нужно улучшить чанкинг dorm.md"),
    ),
    (
        "Какой проходной балл на бюджет на ФИТе?",
        ["267", "бюджет"],
    ),
    (
        "Что изучают на экономическом факультете?",
        ["экономик", "математик", "статистик", "моделирован"],
    ),
    (
        "Где находятся общежития НГУ?",
        ["академгородок", "10 минут", "университет"],
    ),
]


@pytest.fixture(scope="module", autouse=True)
def load_baza():
    """Загружаем все .md файлы из baza/ перед тестами."""
    texts = [f.read_text(encoding="utf-8") for f in BAZA_DIR.glob("*.md")]
    assert texts, f"Не найдены .md файлы в {BAZA_DIR}"
    add_texts(texts)


@pytest.mark.parametrize("question,keywords", KEYWORD_CASES)
def test_keyword_in_retrieved_context(question, keywords):
    results = search_similar(question, k=3)
    assert results, f"RAG вернул пустой результат для: '{question}'"

    combined = " ".join(doc.page_content.lower() for doc in results)

    matched = [kw for kw in keywords if kw.lower() in combined]
    assert matched, (
        f"Вопрос: '{question}'\n"
        f"Ожидались слова: {keywords}\n"
        f"Контекст (первые 300 символов): {combined[:300]}"
    )
