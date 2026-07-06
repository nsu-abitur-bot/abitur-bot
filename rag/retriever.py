from typing import Literal

from rag.graph_memory import get_graph_memory

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
) -> tuple[str, list[dict]]:
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
) -> tuple[str, list[dict]]:
    """CRAG-вариант ретрива: фильтрует чанки до генерации финального ответа."""
    graph_memory = get_graph_memory()
    return await graph_memory.query_with_crag(
        DEFAULT_GRAPH_ID,
        query,
        mode=mode,
        conversation_history=conversation_history,
    )
