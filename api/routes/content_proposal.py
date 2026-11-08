from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import require_admin
from api.services.rag_import_queue import ACTIVE_IMPORT_STATES
from db.postgres.db import get_async_session
from db.postgres.models import Admin, ContentProposal, timestamp
from db.postgres.services.content_proposal import ContentProposalService
from rag.loader import DEFAULT_GRAPH_ID

router = APIRouter(prefix="/content-proposals", tags=["Content proposals"])


class ProposalCreate(BaseModel):
    source: Literal["model", "parser"]
    graph_id: str = Field(default=DEFAULT_GRAPH_ID, min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1)
    source_url: str | None = Field(default=None, max_length=1000)
    source_hash: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")


class ProposalResponse(BaseModel):
    id: str
    source: str
    graph_id: str
    title: str
    text: str
    source_url: str | None
    source_hash: str | None
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


@router.post("/{proposal_id}/approve", status_code=202, response_model=ProposalResponse)
async def approve_proposal(
    proposal_id: str,
    session: AsyncSession = Depends(get_async_session),
    editor: Admin = Depends(require_admin),
) -> ProposalResponse:
    service = ContentProposalService(session)
    proposal = await service.lock_pending(proposal_id)
    if proposal is None:
        existing = await service.get(proposal_id)
        if existing is not None and existing.status in (
            *ACTIVE_IMPORT_STATES,
            "approved",
        ):
            return ProposalResponse.from_record(existing)
        raise HTTPException(status_code=409, detail="Proposal is not pending")

    proposal.status = "queued"
    proposal.reviewer_id = editor.id
    proposal.reviewed_at = timestamp()
    await session.commit()
    return ProposalResponse.from_record(proposal)
