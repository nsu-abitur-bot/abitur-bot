from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.schemas.rag import RagUploadResponse
from api.services.rag_upload import RagUploadService
from rag.loader import DEFAULT_GRAPH_ID

router = APIRouter(prefix="/rag", tags=["RAG Management"])


def get_rag_upload_service() -> RagUploadService:
    return RagUploadService()


@router.post(
    "/upload", response_model=RagUploadResponse, summary="Загрузить документы в RAG"
)
async def upload_documents_to_rag(
    files: list[UploadFile] = File(..., description="Файлы для индексации в RAG"),
    service: RagUploadService = Depends(get_rag_upload_service),
) -> RagUploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        results = await service.ingest_files(files=files, graph_id=DEFAULT_GRAPH_ID)
    except Exception:
        raise HTTPException(status_code=503, detail="RAG ingestion unavailable")

    indexed_count = sum(1 for result in results if result.status == "indexed")
    skipped_count = len(results) - indexed_count

    return RagUploadResponse(
        accepted_formats=service.accepted_formats,
        indexed_count=indexed_count,
        skipped_count=skipped_count,
        results=results,
    )
