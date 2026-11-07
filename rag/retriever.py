from typing import Literal

from rag.graph_memory import RetrievalError, get_graph_memory
from rag.trace import RetrievalResult

__all__ = [
    "DEFAULT_GRAPH_ID",
    "RetrievalError",
    "RetrievalResult",
    "query_graph",
    "query_graph_with_crag",
    "query_graph_with_sources",
]

DEFAULT_GRAPH_ID = "abitur_kb"


async def query_graph(
    query: str,
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "hybrid",
) -> str:
    graph_memory = get_graph_memory()
    return await graph_memory.query(DEFAULT_GRAPH_ID, query, mode=mode)


async def query_graph_with_sources(
    query: str,
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "hybrid",
    conversation_history: str | None = None,
) -> RetrievalResult:
    graph_memory = get_graph_memory()
    return await graph_memory.query_with_sources(
        DEFAULT_GRAPH_ID,
        query,
        mode=mode,
        conversation_history=conversation_history,
    )


async def query_graph_with_crag(
    query: str,
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "hybrid",
    conversation_history: str | None = None,
    search_query: str | None = None,
) -> RetrievalResult:
    """CRAG-вариант ретрива: фильтрует чанки до генерации финального ответа."""
    graph_memory = get_graph_memory()
    return await graph_memory.query_with_crag(
        DEFAULT_GRAPH_ID,
        query,
        mode=mode,
        conversation_history=conversation_history,
        search_query=search_query,
    )
