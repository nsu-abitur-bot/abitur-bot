from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from api.main import app
from api.routes import rag
from api.routes.rag import get_rag_upload_service
from api.schemas.rag import UploadedDocumentResult
from api.services.rag_upload import RagUploadService


class TestRagUploadEndpoint:
    def setup_method(self):
        self.client = TestClient(app)
        self.mock_service = AsyncMock()
        self.mock_service.accepted_formats = [".csv", ".json", ".md", ".txt"]
        app.dependency_overrides[get_rag_upload_service] = lambda: self.mock_service

    def teardown_method(self):
        app.dependency_overrides.pop(get_rag_upload_service, None)

    def test_upload_documents_success(self):
        self.mock_service.ingest_files.return_value = [
            UploadedDocumentResult(
                filename="example.md",
                status="indexed",
                message="Document indexed in RAG",
                chars=42,
            )
        ]

        response = self.client.post(
            "/api/v1/rag/upload",
            files=[("files", ("example.md", "# Title\ncontent", "text/markdown"))],
        )

        assert response.status_code == 200
        data = response.json()
        assert data["indexed_count"] == 1
        assert data["skipped_count"] == 0
        assert data["results"][0]["filename"] == "example.md"

    def test_upload_documents_unsupported(self):
        self.mock_service.ingest_files.return_value = [
            UploadedDocumentResult(
                filename="file.exe",
                status="skipped",
                message="Unsupported format",
                chars=0,
            )
        ]

        response = self.client.post(
            "/api/v1/rag/upload",
            files=[("files", ("file.exe", "binary-data", "application/octet-stream"))],
        )

        assert response.status_code == 200
        data = response.json()
        assert data["indexed_count"] == 0
        assert data["skipped_count"] == 1
        assert data["results"][0]["status"] == "skipped"


class TestRagPreprocessEndpoint:
    def setup_method(self):
        self.client = TestClient(app)

    def test_preprocess_document_success(self, monkeypatch):
        async def fake_clean_and_structure_text(text: str) -> str:
            assert text == "Документы принимает НГУ."
            return (
                "Документы принимает НГУ "
                "(Новосибирский государственный университет)."
            )

        monkeypatch.setattr(
            rag, "clean_and_structure_text", fake_clean_and_structure_text
        )

        response = self.client.post(
            "/api/v1/rag/preprocess",
            json={"text": "  Документы принимает НГУ.  "},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == (
            "Документы принимает НГУ "
            "(Новосибирский государственный университет)."
        )
        assert data["chars"] == len(data["text"])

    def test_preprocess_document_empty_text(self):
        response = self.client.post(
            "/api/v1/rag/preprocess",
            json={"text": "   "},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Text must not be empty"


@pytest.fixture
def rag_service():
    return RagUploadService()


class TestRagUploadService:
    @pytest.mark.asyncio
    async def test_ingest_files_exceeds_size_limit(self, rag_service: RagUploadService):
        with patch.object(
            rag_service,
            "_get_settings",
            AsyncMock(return_value={"max_document_size": "10"}),
        ):
            mock_file = MagicMock(spec=UploadFile)
            mock_file.filename = "large_file.txt"
            mock_file.read = AsyncMock(
                return_value=b"This text is definitely longer than 10 bytes."
            )

            results = await rag_service.ingest_files([mock_file], graph_id="test_graph")

            assert len(results) == 1
            assert results[0].status == "skipped"
            assert "File is too large" in results[0].message

    @pytest.mark.asyncio
    async def test_ingest_files_pdf_page_limit(self, rag_service: RagUploadService):
        with patch.object(
            rag_service,
            "_get_settings",
            AsyncMock(return_value={"max_document_pages": "1"}),
        ):
            mock_file = MagicMock(spec=UploadFile)
            mock_file.filename = "test.pdf"
            mock_file.read = AsyncMock(return_value=b"fake pdf content")

            with patch.object(
                rag_service,
                "_extract_text",
                AsyncMock(
                    side_effect=ValueError("PDF exceeds maximum allowed pages (1)")
                ),
            ):
                results = await rag_service.ingest_files(
                    [mock_file], graph_id="test_graph"
                )

                assert len(results) == 1
                assert results[0].status == "skipped"
                assert "PDF exceeds maximum allowed pages" in results[0].message
