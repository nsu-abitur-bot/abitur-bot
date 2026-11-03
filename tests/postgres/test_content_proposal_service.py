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
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    from api.routes import content_proposal as route
    from db.postgres.models import Admin

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
        return await kwargs["document_service"].create_or_update_for_source(
            graph_id=kwargs["graph_id"],
            title=kwargs["title"],
            source_url=kwargs["source_url"],
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
