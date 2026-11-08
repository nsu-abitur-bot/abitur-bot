import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from api.routes.rag import confirm_rag_upload, get_rag_import, list_rag_documents
from api.schemas.rag import ConfirmUploadRequest
from api.services import rag_import_queue
from api.services.rag_import_queue import enqueue_import
from db.postgres.models import ContentProposal
from db.postgres.services.document import DocumentService
from rag.loader import DEFAULT_GRAPH_ID


@pytest.mark.asyncio
async def test_confirmed_import_is_persisted_and_retry_reuses_job(session):
    args = {
        "title": "Институт",
        "source_url": "https://example.org/institute",
        "content": "Описание программ",
    }
    first = await enqueue_import(session, **args)
    retry = await enqueue_import(session, **args)
    edited = await enqueue_import(session, **(args | {"content": "Новая редакция"}))

    assert first.status == "queued"
    assert first.text == args["content"]
    assert retry.id == first.id
    assert edited.id != first.id


@pytest.mark.asyncio
async def test_worker_finishes_confirmed_import_and_keeps_editor_drafts(
    session, test_engine, monkeypatch
):
    monkeypatch.setattr(
        rag_import_queue,
        "AsyncSessionLocal",
        async_sessionmaker(test_engine, expire_on_commit=False),
    )

    async def finish_indexing(*, document, document_service, new_rag_doc_id, **kwargs):
        materials = await list_rag_documents(session)
        assert {doc.title: doc.import_state for doc in materials.documents} == {
            "Институт": "processing",
            "Черновик": "review",
        }
        await session.commit()
        assert document.rag_doc_id == new_rag_doc_id
        assert document.status == "pending"
        return await document_service.mark_indexed(
            document.id,
            content_hash=kwargs["content_hash"],
            content_length=len(kwargs["text"]),
            rag_doc_id=new_rag_doc_id,
            source_proposal_id=kwargs["source_proposal_id"],
        )

    monkeypatch.setattr(rag_import_queue, "index_document", finish_indexing)
    draft = ContentProposal(
        source="parser", graph_id=DEFAULT_GRAPH_ID, title="Черновик", text="Проверить"
    )
    session.add(draft)
    await session.commit()

    response = await confirm_rag_upload(
        ConfirmUploadRequest(title="Институт", text="Описание", url="", documents=[]),
        session,
    )
    job_id = response["job_id"]
    assert response["status"] == "queued"
    await session.commit()
    assert await rag_import_queue.process_next_import()
    assert not await rag_import_queue.process_next_import()

    session.expire_all()
    assert (await get_rag_import(job_id, session))["status"] == "indexed"
    document = await DocumentService(session).get_by_source_proposal_id(job_id)
    assert document is not None
    assert document.rag_doc_id == job_id
    materials = await list_rag_documents(session)
    assert {doc.title: doc.import_state for doc in materials.documents} == {
        "Институт": None,
        "Черновик": "review",
    }
    await session.refresh(draft)
    assert draft.status == "pending"


@pytest.mark.asyncio
async def test_queued_upload_appears_in_materials_before_worker(session):
    response = await confirm_rag_upload(
        ConfirmUploadRequest(title="Институт", text="Описание", url="", documents=[]),
        session,
    )

    materials = await list_rag_documents(session)
    assert [(doc.title, doc.import_state) for doc in materials.documents] == [
        ("Институт", "queued")
    ]
    assert materials.documents[0].id == response["job_id"]
