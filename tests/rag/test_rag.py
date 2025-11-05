from rag.loader import add_texts
from rag.retriever import search_similar


def test_rag_pipeline():
    add_texts(
        [
            "LangChain используется для RAG-приложений.",
            "Волга впадает в Каспийское море.",
        ]
    )
    results = search_similar("Что такое LangChain?")
    assert len(results) > 0

    top_result = results[0].page_content.lower()
    assert "langchain" in top_result, (
        f"Expected LangChain in top result, got: {top_result}"
    )
