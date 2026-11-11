import builtins
import json
from datetime import datetime

import pytest
from sqlalchemy import event

from api.routes import rag as rag_routes
from db.postgres.models import ContentProposal, Document
from rag.graph_memory import GraphMemory


@pytest.mark.asyncio
async def test_materials_page_filters_merged_queue_and_returns_processing_in_one_read(
    session, test_engine, monkeypatch, tmp_path
):
    memory = GraphMemory()
    monkeypatch.setattr(memory, "_get_workspace_path", lambda _: str(tmp_path))
    monkeypatch.setattr(rag_routes, "get_graph_memory", lambda: memory)
    documents = [
        Document(
            id=f"doc-{i:02}",
            title=f"Материал {i:02}",
            graph_id="abitur_kb",
            rag_doc_id=f"rag-{i}",
            created_at=datetime(2026, 11, 11),
        )
        for i in range(25)
    ]
    session.add_all(documents)
    session.add(
        ContentProposal(
            id="job",
            source="parser",
            graph_id="abitur_kb",
            title="Свежая загрузка",
            text="Текст",
            status="queued",
        )
    )
    await session.commit()
    (tmp_path / "kv_store_doc_status.json").write_text(
        json.dumps(
            {
                f"rag-{i}": {"status": "processed" if i % 2 else "failed"}
                for i in range(25)
            }
        )
    )
    # List status must not read full documents/graph diagnostics.
    (tmp_path / "kv_store_full_docs.json").mkdir()
    reads = []
    original_open = builtins.open

    def track_open(file, *args, **kwargs):
        reads.append(str(file))
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", track_open)
    statements = []

    def track_query(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)

    event.listen(test_engine.sync_engine, "before_cursor_execute", track_query)
    first = await rag_routes.list_rag_documents(session, limit=10, state="ready")
    assert reads == [str(tmp_path / "kv_store_doc_status.json")]
    assert len(statements) == 2
    second = await rag_routes.list_rag_documents(
        session, limit=10, offset=10, state="ready"
    )
    assert first.total == 12 and first.total_all == 26
    assert first.has_pending
    assert len(first.documents) == 10 and len(second.documents) == 2
    assert all(doc.lightrag_status == "processed" for doc in first.documents)
    assert not {d.id for d in first.documents} & {d.id for d in second.documents}
    job = await rag_routes.list_rag_documents(
        session, limit=10, q="загрузка", state="attention"
    )
    assert [doc.id for doc in job.documents] == ["job"]
    assert job.documents[0].proposal_only
    assert (
        await rag_routes.list_rag_documents(session, limit=10, offset=100)
    ).documents == []


@pytest.mark.asyncio
async def test_pending_flag_includes_jobs_hidden_by_source_dedup(session, monkeypatch):
    class Memory:
        async def get_list_docs(self, *args, **kwargs):
            return []

    monkeypatch.setattr(rag_routes, "get_graph_memory", Memory)
    for id, state, day in [("queued", "queued", 1), ("failed", "index_failed", 2)]:
        session.add(
            ContentProposal(
                id=id,
                source="parser",
                graph_id="abitur_kb",
                title=id,
                text="text",
                source_url="https://example.org/shared",
                status=state,
                created_at=datetime(2026, 11, day),
            )
        )
    await session.commit()
    result = await rag_routes.list_rag_documents(session, limit=1)
    assert result.total_all == 1
    assert result.documents[0].id == "failed"
    assert result.has_pending
