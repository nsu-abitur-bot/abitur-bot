"""Real PostgreSQL + LightRAG regression for failed overlapping replacements."""

import pytest
from lightrag.base import DocStatus
from lightrag.utils import GRAPH_FIELD_SEP, compute_mdhash_id
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from api.services.document_indexer import index_document
from db.postgres.models import Document
from db.postgres.services.document import DocumentService
from rag.graph_memory import GraphMemory
from tests.rag.test_document_indexer_integration import (  # noqa: F401
    FakeOpenAILLM,
    clean_lightrag_pipeline_status,
    real_lightrag_env,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["failed", "partial", "identical"])
async def test_overlap_preserves_active_ownership(
    test_engine: AsyncEngine,
    real_lightrag_env: tuple[GraphMemory, FakeOpenAILLM],  # noqa: F811
    replacement: str,
) -> None:
    memory, llm = real_lightrag_env
    llm.extraction_response = (
        "entity<|#|>NSU<|#|>organization<|#|>NSU admits applicants.\n"
        "entity<|#|>Applicants<|#|>category<|#|>Applicants submit documents.\n"
        "relation<|#|>NSU<|#|>Applicants<|#|>admissions<|#|>NSU admits applicants.\n"
        "<|COMPLETE|>"
    )
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    graph_id = "failed_overlap"
    shared = "General applicant information and admission procedures. " * 180
    old_text = shared + "Old residence rules and paper applications. " * 180
    new_text = shared + "Updated residence rules and digital applications. " * 180
    try:
        async with sessions() as session:
            service = DocumentService(session)
            doc = await service.create_or_update_for_source(
                graph_id=graph_id,
                title="Rules",
                source_url="https://example.invalid/rules",
            )
            assert await index_document(
                text=old_text, document=doc, document_service=service, graph_id=graph_id
            )
            old_id = doc.rag_doc_id
            assert old_id
            old_chunks = await memory.get_chunks_list(graph_id, old_id)
            assert len(old_chunks) > 1
            async with memory._use_graph(graph_id) as rag:
                before_nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
                before_edges = await rag.chunk_entity_relation_graph.get_all_edges()
                assert before_nodes and before_edges
                assert rag.tokenizer is not None
                candidate_chunks = {
                    compute_mdhash_id(chunk["content"], prefix="chunk-")
                    for chunk in rag.chunking_func(
                        rag.tokenizer,
                        new_text,
                        None,
                        False,
                        rag.chunk_overlap_token_size,
                        rag.chunk_token_size,
                    )
                }
                assert candidate_chunks & set(old_chunks)
                assert candidate_chunks - set(old_chunks)
            llm.should_fail = replacement == "failed"
            result = await index_document(
                text=old_text if replacement == "identical" else new_text,
                document=doc,
                document_service=service,
                graph_id=graph_id,
            )
            assert (result is None) == (replacement == "failed")
            active_id = doc.rag_doc_id
            assert active_id
            assert (active_id == old_id) == (replacement == "failed")
            await session.commit()
            async with sessions() as reader:
                persisted = (await reader.execute(select(Document))).scalar_one()
                assert persisted.rag_doc_id == active_id
            assert await memory.get_doc_status(graph_id, active_id) == "processed"
            active_chunks = await memory.get_chunks_list(graph_id, active_id)
            assert set(old_chunks) & set(active_chunks)
            if replacement == "partial":
                assert set(old_chunks) - set(active_chunks)
                assert set(active_chunks) - set(old_chunks)
            # Verify persisted stores through a new LightRAG instance.
            await memory.cleanup()
            async with memory._use_graph(graph_id) as rag:
                chunks = await rag.text_chunks.get_by_ids(active_chunks)
                assert all(chunks), "Failed cleanup removed active text chunks"
                vectors = await rag.chunks_vdb.get_by_ids(active_chunks)
                assert all(vectors), "Failed cleanup removed active chunk vectors"
                assert all(chunk["full_doc_id"] == active_id for chunk in chunks)
                assert all(vector["full_doc_id"] == active_id for vector in vectors)
                obsolete_chunks = (
                    candidate_chunks - set(old_chunks)
                    if replacement == "failed"
                    else set(old_chunks) - set(active_chunks)
                )
                assert not any(await rag.text_chunks.get_by_ids(list(obsolete_chunks)))
                assert not any(await rag.chunks_vdb.get_by_ids(list(obsolete_chunks)))
                assert not await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
                if replacement != "failed":
                    assert await rag.full_entities.get_by_id(old_id) is None
                    assert await rag.full_relations.get_by_id(old_id) is None
                nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
                edges = await rag.chunk_entity_relation_graph.get_all_edges()
                assert len(nodes) == len(before_nodes)
                assert len(edges) == len(before_edges)
                for item in nodes + edges:
                    sources = set(item["source_id"].split(GRAPH_FIELD_SEP))
                    assert sources and sources <= set(active_chunks)
                entity_ids = [
                    compute_mdhash_id(node["id"], prefix="ent-") for node in nodes
                ]
                assert all(await rag.entities_vdb.get_by_ids(entity_ids))
                for edge in edges:
                    source, target = sorted((edge["source"], edge["target"]))
                    relation_id = compute_mdhash_id(source + target, prefix="rel-")
                    assert await rag.relationships_vdb.get_by_id(relation_id)
    finally:
        await memory.cleanup()
