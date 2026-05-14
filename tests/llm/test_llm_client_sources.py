from llm.llm_client import _extract_sources_from_rag_context


def test_extract_sources_from_rag_context():
    context = """
    Ответ LightRAG.
    <a href="https://example.test/admission">Прием</a>
    [Общежития](https://example.test/dorms)
    Источник информации (https://example.test/docs/rules%20(2).pdf).
    Повтор: https://example.test/admission
    """

    sources = _extract_sources_from_rag_context(context)

    assert sources == [
        {"url": "https://example.test/admission", "title": "Прием"},
        {"url": "https://example.test/dorms", "title": "Общежития"},
        {
            "url": "https://example.test/docs/rules%20(2).pdf",
            "title": "example.test",
        },
    ]


def test_extract_sources_from_lightrag_references_block():
    context = """
    Собрания для первокурсников проводятся деканатами перед началом учебного года.

    ### References

    - [2] https://education.nsu.ru/freshmen/
    """

    sources = _extract_sources_from_rag_context(context)

    assert sources == [
        {"url": "https://education.nsu.ru/freshmen/", "title": "Источник 2"}
    ]
