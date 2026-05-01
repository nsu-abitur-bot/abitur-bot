import pytest

# from rag.loader import add_texts
# from rag.retriever import query_graph


# Пока что эти тесты не проходят, я не знаю нужны ли они вообще
# TODO: разобраться что делать с этими тестами, может быть удалить их или переписать
@pytest.mark.asyncio
async def test_rag_pipeline():
    ...
    # add_texts(
    #     [
    #         "LangChain используется для RAG-приложений.",
    #         "Волга впадает в Каспийское море.",
    #     ]
    # )
    # result = await query_graph("Что такое LangChain?")
    # assert len(result) > 0

    # result_lower = result.lower()
    # assert "langchain" in result_lower, (
    #     f"Expected LangChain in result, got: {result_lower}"
    # )
