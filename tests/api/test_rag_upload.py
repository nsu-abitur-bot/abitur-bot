from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from api.main import app
from api.routes.rag import get_rag_upload_service
from api.schemas.rag import UploadedDocumentResult


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
