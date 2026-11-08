"""Durable queue for pages an administrator has confirmed for indexing."""

import asyncio
import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.services.document_indexer import index_document
from api.services.document_update import calculate_content_hash, fetch_url_bytes
from db.postgres.db import AsyncSessionLocal
from db.postgres.models import ContentProposal
from db.postgres.services.document import DocumentService
from rag.loader import DEFAULT_GRAPH_ID

logger = logging.getLogger(__name__)

IMPORT_STATES = ("queued", "indexed", "index_failed")
_WORKER_LOCK = 726_414_720


async def enqueue_import(
    session: AsyncSession, *, title: str, source_url: str | None, content: str
) -> ContentProposal:
    key = f"rag-import:{DEFAULT_GRAPH_ID}:{source_url or title}"
    await session.execute(
        select(func.pg_advisory_xact_lock(func.hashtextextended(key, 0)))
    )
    same_source = ContentProposal.source_url == source_url
    if source_url is None:
        same_source &= ContentProposal.title == title
    previous = await session.scalar(
        select(ContentProposal)
        .where(
            ContentProposal.graph_id == DEFAULT_GRAPH_ID,
            ContentProposal.status.in_(IMPORT_STATES),
            same_source,
        )
        .order_by(ContentProposal.created_at.desc(), ContentProposal.id.desc())
        .limit(1)
    )
    if (
        previous is not None
        and previous.status == "queued"
        and previous.title == title
        and previous.text == content
    ):
        return previous

    job = ContentProposal(
        source="parser",
        graph_id=DEFAULT_GRAPH_ID,
        status="queued",
        title=title,
        source_url=source_url,
        text=content,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def process_next_import() -> bool:
    async with AsyncSessionLocal() as queue_session, queue_session.begin():
        acquired = await queue_session.scalar(
            select(func.pg_try_advisory_xact_lock(_WORKER_LOCK))
        )
        if not acquired:
            return False
        job = await queue_session.scalar(
            select(ContentProposal)
            .where(ContentProposal.status == "queued")
            .order_by(ContentProposal.created_at, ContentProposal.id)
            .limit(1)
        )
        if job is None:
            return False

        content_hash = calculate_content_hash(job.text.encode("utf-8"))
        if job.source_url:
            try:
                content_hash = calculate_content_hash(
                    await fetch_url_bytes(job.source_url)
                )
            except Exception:
                logger.warning("Could not fetch source hash for RAG import %s", job.id)

        async with AsyncSessionLocal() as session:
            service = DocumentService(session)
            document = await service.get_by_source_proposal_id(job.id)
            if document is None:
                document = await service.create_or_update_for_source(
                    graph_id=job.graph_id,
                    title=job.title,
                    source_url=job.source_url,
                    rag_doc_id=job.id,
                    source_proposal_id=job.id,
                )
            # Long LightRAG calls should only hold the worker lock connection.
            await session.commit()
            indexed = await index_document(
                document=document,
                title=job.title,
                source_url=job.source_url,
                graph_id=job.graph_id,
                text=job.text,
                content_hash=content_hash,
                document_service=service,
                source_proposal_id=job.id,
                new_rag_doc_id=job.id,
            )

        job.status = "indexed" if indexed is not None else "index_failed"
        job.document_id = indexed.id if indexed is not None else document.id
        return True


async def run_import_queue() -> None:
    while True:
        try:
            if await process_next_import():
                continue
        except Exception:
            logger.exception("RAG import queue processing failed")
        await asyncio.sleep(2)
