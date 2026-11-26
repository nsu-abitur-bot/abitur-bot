"""Issue #481 through Redis, an indexed LightRAG workspace and real CRAG filters."""

import asyncio
import logging
from functools import partial

import numpy as np
import pytest
from fakeredis.aioredis import FakeRedis

from db.redis.client import RedisClient
from evals.dialog_scope import (
    CHUNKS,
    DEADLINE,
    DEADLINE_URL,
    DOCUMENT,
    DOCUMENT_URL,
    DOCUMENTS_QUESTION,
    FACULTY_QUESTION,
    MASTER_DEADLINE,
    MASTER_URL,
    OLD_QUESTION,
    OTHER_YEAR_DEADLINE,
    OTHER_YEAR_URL,
    PAID_DEADLINE,
    PAID_URL,
    PORTAL,
    PORTAL_QUESTION,
    PORTAL_URL,
    QUESTION,
    DialogProvider,
)
from pipeline import llm_client
from rag import crag, graph_memory, retriever
from tests.rag.test_crag import crag_engine as crag_engine
from tests.rag.test_crag import crag_sessionmaker as crag_sessionmaker
from tests.rag.test_crag import seeded as seeded
from tests.rag.test_retrieval_failures import (
    EMBEDDING_DIM,
    _EmbeddingSwitch,
    _make_memory,
)


class _FixedEmbedding(_EmbeddingSwitch):
    async def __call__(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), EMBEDDING_DIM))


@pytest.mark.asyncio
async def test_stale_dialog_does_not_hide_indexed_facts(
    seeded,  # noqa: F811
    tmp_path,
    monkeypatch,
    caplog,
):
    memory, index = await _make_memory(tmp_path, _FixedEmbedding())
    await index.ainsert(
        [chunk["content"] for chunk in CHUNKS],
        file_paths=[chunk["file_path"] for chunk in CHUNKS],
    )
    searches = []
    filtered = []
    generation = []
    retrieve_raw = memory._retrieve_raw_chunks
    run_crag = graph_memory.run_crag
    generate = DialogProvider.generate_with_tools

    async def capture_search(rag, query, mode):
        chunks = await retrieve_raw(rag, query, mode)
        searches.append((query, [chunk["file_path"] for chunk in chunks]))
        return chunks

    async def capture_crag(question, chunks, **kwargs):
        kept, hint = await run_crag(question, chunks, **kwargs)
        filtered.append(
            (
                question,
                hint.faculty.name if hint.faculty else None,
                hint.level,
                [chunk.source_url for chunk in kept],
            )
        )
        return kept, hint

    async def capture_generation(provider, messages, **kwargs):
        generation.append([str(message.content) for message in messages[1:]])
        return await generate(provider, messages, **kwargs)

    async def config():
        return crag.CragConfig(
            enabled=True,
            relevance_threshold=0.5,
            min_chunks=2,
            allow_refine=True,
            use_faculty_table=True,
            max_graded_chunks=12,
        )

    async def no_faq(_matcher, _question):
        return ""

    store = object.__new__(RedisClient)
    async with FakeRedis(decode_responses=True) as redis:
        store.client = redis

        async def get_redis():
            return store

        monkeypatch.setattr(memory, "_retrieve_raw_chunks", capture_search)
        monkeypatch.setattr(graph_memory, "run_crag", capture_crag)
        monkeypatch.setattr(retriever, "get_graph_memory", lambda: memory)
        monkeypatch.setattr(graph_memory, "load_crag_config", config)
        monkeypatch.setattr(graph_memory, "AsyncSessionLocal", seeded)
        monkeypatch.setattr(llm_client, "AsyncSessionLocal", seeded)
        monkeypatch.setattr(llm_client, "load_crag_config", config)
        monkeypatch.setattr(llm_client, "get_redis_client", get_redis)
        monkeypatch.setattr(llm_client, "_expand_abbrevs", lambda question, _: question)
        monkeypatch.setattr(llm_client, "_match_faq", no_faq)
        monkeypatch.setattr(llm_client, "get_faq_matcher", lambda: None)
        monkeypatch.setattr(
            llm_client,
            "query_graph_with_crag",
            partial(retriever.query_graph_with_crag, mode="naive"),
        )
        monkeypatch.setattr(llm_client, "get_llm_provider", DialogProvider)
        monkeypatch.setattr(crag, "get_llm_provider", DialogProvider)
        monkeypatch.setattr(DialogProvider, "generate_with_tools", capture_generation)
        caplog.set_level(logging.INFO, logger="rag.crag")

        async def ask(question, session):
            search_start = len(searches)
            answer = await llm_client.ask_local_llm(question, session)
            await asyncio.gather(*llm_client._background_tasks)
            snapshot = (
                searches[search_start:],
                filtered[-1],
                generation[-1],
                answer.text,
                [source["url"] for source in answer.sources],
            )
            print(snapshot)
            return snapshot

        try:
            old = await ask(OLD_QUESTION, "dialog")
            stale = await ask(QUESTION, "dialog")
            paid = await ask("А на платное?", "dialog")
            other_year = await ask("А в другом году?", "dialog")
            follow_up = await ask(DOCUMENTS_QUESTION, "dialog")
            faculty = await ask(FACULTY_QUESTION, "dialog")
            portal = await ask(PORTAL_QUESTION, "dialog")
            fresh = await ask(QUESTION, "fresh")
        finally:
            await index.finalize_storages()

    assert "Информация не найдена" in old[3]
    assert (
        set(stale[0][0][1])
        == set(fresh[0][0][1])
        == {chunk["file_path"] for chunk in CHUNKS}
    )
    assert DEADLINE in fresh[3] and fresh[4] == [DEADLINE_URL]
    assert DEADLINE in stale[3] and stale[4] == [DEADLINE_URL], stale
    assert stale[1] == fresh[1]
    assert stale[0] == fresh[0]
    assert stale[2] == [QUESTION]
    assert DOCUMENT in follow_up[3] and follow_up[4] == [DOCUMENT_URL], follow_up
    assert all("ФИТ" not in query for query, _ in follow_up[0])
    assert all("Информация не найдена" not in message for message in follow_up[2])
    assert PAID_DEADLINE in paid[3] and paid[4] == [PAID_URL], paid
    assert OTHER_YEAR_DEADLINE in other_year[3] and other_year[4] == [OTHER_YEAR_URL]
    assert PORTAL in portal[3] and portal[4] == [PORTAL_URL], portal
    assert portal[1][1:3] == (None, "bachelor")
    assert portal[2] == [PORTAL_QUESTION]
    assert "ММФ" not in portal[0][0][0]
    assert MASTER_DEADLINE in faculty[3] and faculty[4] == [MASTER_URL], faculty
    assert faculty[1][1:3] == ("Механико-математический факультет", "master")
    assert "Бизнес-информатика" not in faculty[0][0][0]
