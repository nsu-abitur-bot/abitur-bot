from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import require_admin
from api.services.document_indexer import index_document
from api.services.document_update import calculate_content_hash
from db.postgres.db import get_async_session
from db.postgres.models import Admin, ContentProposal
from db.postgres.services.content_proposal import ContentProposalService
from db.postgres.services.document import DocumentService
from rag.loader import DEFAULT_GRAPH_ID

router = APIRouter(prefix="/content-proposals", tags=["Content proposals"])


class ProposalCreate(BaseModel):
    source: Literal["model", "parser"]
    graph_id: str = Field(default=DEFAULT_GRAPH_ID, min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1)
    source_url: str | None = Field(default=None, max_length=1000)


class ProposalResponse(BaseModel):
    id: str
    source: str
    graph_id: str
    title: str
    text: str
    source_url: str | None
    status: str
    reviewer_id: str | None
    document_id: str | None
    created_at: datetime
    reviewed_at: datetime | None

    @classmethod
    def from_record(cls, proposal: ContentProposal) -> "ProposalResponse":
        return cls.model_validate(proposal, from_attributes=True)


@router.post("", status_code=201, response_model=ProposalResponse)
async def create_proposal(
    item: ProposalCreate,
    session: AsyncSession = Depends(get_async_session),
) -> ProposalResponse:
    service = ContentProposalService(session)
    try:
        proposal = await service.create(**item.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ProposalResponse.from_record(proposal)


@router.get("", response_model=list[ProposalResponse])
async def list_proposals(
    session: AsyncSession = Depends(get_async_session),
) -> list[ProposalResponse]:
    proposals = await ContentProposalService(session).list_pending()
    return [ProposalResponse.from_record(proposal) for proposal in proposals]


@router.post("/{proposal_id}/reject", response_model=ProposalResponse)
async def reject_proposal(
    proposal_id: str,
    session: AsyncSession = Depends(get_async_session),
    editor: Admin = Depends(require_admin),
) -> ProposalResponse:
    proposal = await ContentProposalService(session).reject(proposal_id, editor.id)
    if proposal is None:
        raise HTTPException(status_code=409, detail="Proposal is not pending")
    return ProposalResponse.from_record(proposal)


@router.post("/{proposal_id}/approve", response_model=ProposalResponse)
async def approve_proposal(
    proposal_id: str,
    session: AsyncSession = Depends(get_async_session),
    editor: Admin = Depends(require_admin),
) -> ProposalResponse:
    service = ContentProposalService(session)
    proposal = await service.claim(proposal_id)
    if proposal is None:
        raise HTTPException(status_code=409, detail="Proposal is not pending")

    try:
        document_service = DocumentService(session)
        document = (
            await document_service.get_by_source_url(
                proposal.graph_id, proposal.source_url
            )
            if proposal.source_url
            else None
        )
        indexed = await index_document(
            document=document,
            title=proposal.title,
            text=proposal.text,
            source_url=proposal.source_url,
            content_hash=calculate_content_hash(proposal.text.encode("utf-8")),
            graph_id=proposal.graph_id,
            session=session,
            document_service=document_service,
        )
        if indexed is None:
            raise RuntimeError("RAG indexing failed")
    except Exception as exc:
        await session.rollback()
        await service.finish(proposal_id, editor.id, None)
        raise HTTPException(status_code=503, detail="RAG indexing failed") from exc

    reviewed = await service.finish(proposal_id, editor.id, indexed.id)
    return ProposalResponse.from_record(reviewed)
