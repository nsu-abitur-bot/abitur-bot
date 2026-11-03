from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ContentProposal, timestamp


class ContentProposalService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        source: str,
        title: str,
        text: str,
        source_url: str | None = None,
        graph_id: str,
    ) -> ContentProposal:
        if source not in {"model", "parser"}:
            raise ValueError("Invalid proposal source")
        if not title.strip() or not text.strip():
            raise ValueError("Title and text are required")
        if not graph_id.strip():
            raise ValueError("Graph ID is required")
        proposal = ContentProposal(
            source=source,
            graph_id=graph_id,
            title=title.strip(),
            text=text.strip(),
            source_url=source_url,
        )
        self.session.add(proposal)
        await self.session.commit()
        await self.session.refresh(proposal)
        return proposal

    async def get(self, proposal_id: str) -> ContentProposal | None:
        return await self.session.get(ContentProposal, proposal_id)

    async def list_pending(self) -> list[ContentProposal]:
        result = await self.session.execute(
            select(ContentProposal)
            .where(ContentProposal.status == "pending")
            .order_by(ContentProposal.created_at, ContentProposal.id)
        )
        return list(result.scalars().all())

    async def reject(
        self, proposal_id: str, reviewer_id: str | None
    ) -> ContentProposal | None:
        result = await self.session.execute(
            update(ContentProposal)
            .where(ContentProposal.id == proposal_id, ContentProposal.status == "pending")
            .values(status="rejected", reviewer_id=reviewer_id, reviewed_at=timestamp())
            .returning(ContentProposal.id)
        )
        if result.scalar_one_or_none() is None:
            return None
        await self.session.commit()
        return await self.get(proposal_id)

    async def claim(self, proposal_id: str) -> ContentProposal | None:
        result = await self.session.execute(
            update(ContentProposal)
            .where(ContentProposal.id == proposal_id, ContentProposal.status == "pending")
            .values(status="processing")
            .returning(ContentProposal.id)
        )
        if result.scalar_one_or_none() is None:
            return None
        await self.session.commit()
        return await self.get(proposal_id)

    async def finish(
        self, proposal_id: str, reviewer_id: str, document_id: str | None
    ) -> ContentProposal:
        proposal = await self.get(proposal_id)
        assert proposal is not None
        proposal.status = "approved" if document_id else "pending"
        if document_id:
            proposal.reviewer_id = reviewer_id
            proposal.document_id = document_id
            proposal.reviewed_at = timestamp()
        await self.session.commit()
        await self.session.refresh(proposal)
        return proposal
