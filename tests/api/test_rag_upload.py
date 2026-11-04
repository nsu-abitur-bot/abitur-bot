import io
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from starlette import formparsers
from starlette.datastructures import Headers

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
                status="queued",
                message="Waiting for editor review",
                chars=42,
            )
        ]

        response = self.client.post(
            "/api/v1/rag/upload",
            files=[("files", ("example.md", "# Title\ncontent", "text/markdown"))],
        )

        assert response.status_code == 200
        data = response.json()
        assert data["queued_count"] == 1
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
        assert data["queued_count"] == 0
        assert data["skipped_count"] == 1
        assert data["results"][0]["status"] == "skipped"

    def test_upload_rejects_too_many_files_before_ingestion(self):
        response = self.client.post(
            "/api/v1/rag/upload",
            files=[("files", (f"{i}.txt", b"x", "text/plain")) for i in range(11)],
        )

        assert response.status_code == 400
        self.mock_service.ingest_files.assert_not_awaited()

    def test_upload_rejects_large_content_length_before_parsing(self):
        response = self.client.post(
            "/api/v1/rag/upload",
            files=[("files", ("small.txt", b"x", "text/plain"))],
            headers={"content-length": str(600 * 1024 * 1024)},
        )

        assert response.status_code == 413
        self.mock_service.ingest_files.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("declared_size", [None, "1"])
    async def test_upload_stops_oversized_stream_and_closes_files(
        self, monkeypatch, declared_size
    ):
        monkeypatch.setattr(rag, "max_document_size_bytes", lambda: 1024)
        monkeypatch.setattr(formparsers.MultiPartParser, "spool_max_size", 1024)
        temporary_files = []

        def track_temporary_file(*args, **kwargs):
            file = tempfile.SpooledTemporaryFile(*args, **kwargs)
            temporary_files.append(file)
            return file

        monkeypatch.setattr(formparsers, "SpooledTemporaryFile", track_temporary_file)
        chunks_received = 0
        total_chunks = 34

        async def stream():
            nonlocal chunks_received
            chunks_received += 1
            yield (
                b'--boundary\r\nContent-Disposition: form-data; name="files"; '
                b'filename="large.txt"\r\nContent-Type: text/plain\r\n\r\n'
            )
            for _ in range(32):
                chunks_received += 1
                yield b"x" * (64 * 1024)
            chunks_received += 1
            yield b"\r\n--boundary--\r\n"

        headers = {"content-type": "multipart/form-data; boundary=boundary"}
        if declared_size is not None:
            headers["content-length"] = declared_size
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/rag/upload", content=stream(), headers=headers
            )

        assert response.status_code == 413
        assert response.json()["detail"] == "Upload request is too large"
        assert chunks_received < total_chunks
        assert temporary_files
        assert any(file._rolled for file in temporary_files)
        assert all(file.closed for file in temporary_files)
        self.mock_service.ingest_files.assert_not_awaited()

    def test_upload_is_described_as_multipart_in_openapi(self):
        operation = app.openapi()["paths"]["/api/v1/rag/upload"]["post"]
        schema = operation["requestBody"]["content"]["multipart/form-data"]["schema"]

        assert schema["properties"]["files"]["items"] == {
            "type": "string",
            "format": "binary",
        }


class TestRagPreprocessEndpoint:
    def setup_method(self):
        self.client = TestClient(app)

    def test_preprocess_document_success(self, monkeypatch):
        async def fake_clean_and_structure_text(text: str) -> str:
            assert text == "Документы принимает НГУ."
            return "Документы принимает НГУ (Новосибирский государственный университет)."

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
            "Документы принимает НГУ (Новосибирский государственный университет)."
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
    async def test_ingest_files_rejects_content_length_without_reading(
        self, rag_service: RagUploadService
    ):
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "large.txt"
        mock_file.headers = Headers({"content-length": "11"})
        mock_file.read = AsyncMock()

        with patch("api.services.rag_upload.get_setting", side_effect=["10", "50"]):
            results = await rag_service.ingest_files([mock_file], graph_id="test_graph")

        assert results[0].status == "skipped"
        mock_file.read.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ingest_files_stops_reading_after_limit(
        self, rag_service: RagUploadService
    ):
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "large.txt"
        mock_file.headers = Headers()
        mock_file.read = AsyncMock(side_effect=[b"12345678", b"901"])

        with patch("api.services.rag_upload.get_setting", side_effect=["10", "50"]):
            results = await rag_service.ingest_files([mock_file], graph_id="test_graph")

        assert results[0].status == "skipped"
        assert mock_file.read.await_count == 2
        assert all(call.args[0] <= 11 for call in mock_file.read.await_args_list)

    @pytest.mark.asyncio
    async def test_ingest_files_accepts_exact_size_limit(
        self, rag_service: RagUploadService
    ):
        file = UploadFile(file=io.BytesIO(b"1234567890"), filename="exact.txt")
        with (
            patch("api.services.rag_upload.get_setting", side_effect=["10", "50"]),
            patch.object(
                rag_service,
                "_extract_text",
                AsyncMock(side_effect=ValueError("extraction reached")),
            ) as extract,
        ):
            results = await rag_service.ingest_files([file], graph_id="test_graph")

        assert results[0].message == "extraction reached"
        extract.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ingest_files_exceeds_size_limit(self, rag_service: RagUploadService):
        mock_file = UploadFile(
            file=io.BytesIO(b"This text is definitely longer than 10 bytes."),
            filename="large_file.txt",
        )
        with patch("api.services.rag_upload.get_setting", side_effect=["10", "50"]):
            results = await rag_service.ingest_files([mock_file], graph_id="test_graph")

        assert len(results) == 1
        assert results[0].status == "skipped"
        assert "File is too large" in results[0].message

    @pytest.mark.asyncio
    async def test_ingest_files_pdf_page_limit(self, rag_service: RagUploadService):
        mock_file = UploadFile(file=io.BytesIO(b"fake pdf content"), filename="test.pdf")
        with (
            patch("api.services.rag_upload.get_setting", side_effect=["100", "1"]),
            patch.object(
                rag_service,
                "_extract_text",
                AsyncMock(
                    side_effect=ValueError("PDF exceeds maximum allowed pages (1)")
                ),
            ),
        ):
            results = await rag_service.ingest_files([mock_file], graph_id="test_graph")

        assert len(results) == 1
        assert results[0].status == "skipped"
        assert "PDF exceeds maximum allowed pages" in results[0].message

    @pytest.mark.asyncio
    async def test_ingest_files_queues_extracted_text(
        self, rag_service: RagUploadService, monkeypatch
    ):
        saved = []

        class FakeSessionContext:
            async def __aenter__(self):  # noqa: ANN001
                return object()

            async def __aexit__(self, exc_type, exc, traceback):  # noqa: ANN001
                return None

        class FakeProposalService:
            def __init__(self, session):  # noqa: ANN001
                pass

            async def create(self, **kwargs):  # noqa: ANN001
                saved.append(kwargs)

        from api.services import rag_upload

        monkeypatch.setattr(rag_upload, "AsyncSessionLocal", lambda: FakeSessionContext())
        monkeypatch.setattr(rag_upload, "ContentProposalService", FakeProposalService)

        files = [
            UploadFile(file=io.BytesIO(b"plain text content"), filename="test.txt"),
            UploadFile(file=io.BytesIO(b"second document"), filename="second.txt"),
        ]
        results = await rag_service.ingest_files(files, graph_id="test_graph")

        assert [result.status for result in results] == ["queued", "queued"]
        assert saved == [
            {
                "source": "parser",
                "graph_id": "test_graph",
                "title": "test.txt",
                "source_url": None,
                "text": "plain text content",
            },
            {
                "source": "parser",
                "graph_id": "test_graph",
                "title": "second.txt",
                "source_url": None,
                "text": "second document",
            },
        ]
