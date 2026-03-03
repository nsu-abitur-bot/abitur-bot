from typing import Literal

from rag.graph_memory import get_graph_memory

DEFAULT_GRAPH_ID = "abitur_kb"


async def query_graph(
    query: str,
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "hybrid",
) -> str:
    """
    Выполняет запрос к графовой памяти.

    Args:
        query: Вопрос для запроса.
        mode: Режим запроса ("local", "global", "hybrid", "naive", "mix").

    Returns:
        Сгенерированный ответ.
    """
    graph_memory = get_graph_memory()
    return await graph_memory.query(DEFAULT_GRAPH_ID, query, mode=mode)
