import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.models import Document
from db.postgres.services.content_proposal import ContentProposalService


@pytest.mark.asyncio
async def test_parser_proposal_waits_for_review(session: AsyncSession) -> None:
    service = ContentProposalService(session)
    proposal = await service.create(
        source="parser",
        graph_id="test_graph",
        title="Правила приёма",
        text="Текст правил",
        source_url="https://example.org/rules",
    )

    assert proposal.status == "pending"
    assert await session.scalar(select(Document.id)) is None
    assert [item.id for item in await service.list_pending()] == [proposal.id]

    rejected = await service.reject(proposal.id, reviewer_id=None)
    assert rejected is not None
    assert rejected.status == "rejected"
    assert await service.list_pending() == []
    assert await session.scalar(select(Document.id)) is None


@pytest.mark.asyncio
async def test_model_proposal_is_indexed_only_when_approved(
    session: AsyncSession, test_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from api.routes import content_proposal as route
    from db.postgres.models import Admin

    monkeypatch.setattr(
        route,
        "AsyncSessionLocal",
        async_sessionmaker(test_engine, expire_on_commit=False),
    )
    service = ContentProposalService(session)
    proposal = await service.create(
        source="model",
        graph_id="test_graph",
        title="Сроки подачи",
        text="Подавать документы до указанной даты.",
    )
    editor = Admin(id="editor-id", username="editor", password_hash="unused")
    indexed_texts = []

    async def fake_index_document(**kwargs):  # noqa: ANN003
        indexed_texts.append(kwargs["text"])
        return await kwargs["document_service"].mark_indexed(
            kwargs["document"].id,
            content_hash=kwargs["content_hash"],
            content_length=len(kwargs["text"]),
            title=kwargs["title"],
            source_url=kwargs["source_url"],
            rag_doc_id="new-rag-id",
            source_proposal_id=kwargs["source_proposal_id"],
        )

    monkeypatch.setattr(route, "index_document", fake_index_document)

    assert await session.scalar(select(Document.id)) is None
    reviewed = await route.approve_proposal(proposal.id, session, editor)
    assert reviewed.status == "approved"
    assert reviewed.reviewer_id == editor.id
    assert reviewed.document_id == await session.scalar(select(Document.id))
    assert indexed_texts == ["Подавать документы до указанной даты."]
    assert await service.list_pending() == []

    with pytest.raises(HTTPException) as exc_info:
        await route.approve_proposal(proposal.id, session, editor)
    assert exc_info.value.status_code == 409
    assert len(indexed_texts) == 1


@pytest.mark.asyncio
async def test_cancelled_approval_returns_proposal_to_queue(
    test_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from sqlalchemy.ext.asyncio import async_sessionmaker

    from api.routes import content_proposal as route
    from db.postgres.models import Admin

    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr(route, "AsyncSessionLocal", factory, raising=False)
    async with factory() as session:
        proposal = await ContentProposalService(session).create(
            source="parser", graph_id="test_graph", title="Правила", text="Содержание"
        )

    async def cancelled_index(**kwargs):  # noqa: ANN003
        raise asyncio.CancelledError

    monkeypatch.setattr(route, "index_document", cancelled_index)
    editor = Admin(id="editor-id", username="editor", password_hash="unused")
    async with factory() as session:
        with pytest.raises(asyncio.CancelledError):
            await route.approve_proposal(proposal.id, session, editor)

    async with factory() as session:
        pending = await ContentProposalService(session).list_pending()
        assert [item.id for item in pending] == [proposal.id]
        assert pending[0].status == "pending"


@pytest.mark.asyncio
async def test_retry_recognizes_indexing_completed_before_cancellation(
    test_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from sqlalchemy.ext.asyncio import async_sessionmaker

    from api.routes import content_proposal as route
    from db.postgres.models import Admin

    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr(route, "AsyncSessionLocal", factory)
    async with factory() as session:
        proposal = await ContentProposalService(session).create(
            source="parser", graph_id="test_graph", title="Правила", text="Содержание"
        )

    calls = 0

    async def finish_index_then_cancel(**kwargs):  # noqa: ANN003
        nonlocal calls
        calls += 1
        await kwargs["document_service"].mark_indexed(
            kwargs["document"].id,
            content_hash=kwargs["content_hash"],
            content_length=len(kwargs["text"]),
            rag_doc_id="indexed-rag-id",
            source_proposal_id=kwargs["source_proposal_id"],
        )
        raise asyncio.CancelledError

    monkeypatch.setattr(route, "index_document", finish_index_then_cancel)
    editor = Admin(id="editor-id", username="editor", password_hash="unused")
    async with factory() as session:
        with pytest.raises(asyncio.CancelledError):
            await route.approve_proposal(proposal.id, session, editor)

    async with factory() as session:
        assert [p.id for p in await ContentProposalService(session).list_pending()] == [
            proposal.id
        ]
        reviewed = await route.approve_proposal(proposal.id, session, editor)
        assert reviewed.status == "approved"
        assert calls == 1


@pytest.mark.asyncio
async def test_approval_preserves_raw_source_hash_for_change_detection(
    test_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from sqlalchemy.ext.asyncio import async_sessionmaker

    from api.routes import content_proposal as route
    from api.services import document_update
    from db.postgres.models import Admin
    from db.postgres.services.document import DOCUMENT_STATUS_UNCHANGED

    source_bytes = b"<html><body>Original</body></html>"
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr(route, "AsyncSessionLocal", factory)
    async with factory() as session:
        proposal = await ContentProposalService(session).create(
            source="parser",
            graph_id="test_graph",
            title="Правила",
            text="Parsed text",
            source_url="https://example.org/rules",
            source_hash=source_hash,
        )

    async def fake_index(**kwargs):  # noqa: ANN003
        assert kwargs["content_hash"] == source_hash
        return await kwargs["document_service"].mark_indexed(
            kwargs["document"].id,
            content_hash=kwargs["content_hash"],
            content_length=len(kwargs["text"]),
            rag_doc_id="indexed-rag-id",
            source_proposal_id=kwargs["source_proposal_id"],
        )

    async def fake_fetch(url: str) -> bytes:
        return source_bytes

    monkeypatch.setattr(route, "index_document", fake_index)
    monkeypatch.setattr(document_update, "fetch_url_bytes", fake_fetch)
    editor = Admin(id="editor-id", username="editor", password_hash="unused")
    async with factory() as session:
        reviewed = await route.approve_proposal(proposal.id, session, editor)
        document = await session.get(Document, reviewed.document_id)
        assert document is not None
        assert document.content_hash == source_hash
        check = await document_update.DocumentUpdateService(
            session, graph_id="test_graph"
        )._check_document(document)
        assert check["status"] == DOCUMENT_STATUS_UNCHANGED
