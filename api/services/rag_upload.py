import csv
import io
import json
from collections.abc import Sequence
from html.parser import HTMLParser
from pathlib import Path

from starlette.datastructures import UploadFile

from abbrev.expander import get_abbrev_expander
from api.schemas.rag import UploadedDocumentResult
from api.services.document_update import first_http_url
from config.instance import get_setting
from db.postgres.db import AsyncSessionLocal
from db.postgres.services.content_proposal import ContentProposalService
from parser.pdf import pdf_to_base64_images
from parser.utils import extract_sources
from parser.vision import parse_images_with_llm
from pipeline.preprocessor import generate_title_from_text

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".csv",
    ".html",
    ".htm",
    ".pdf",
}
DEFAULT_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_FILES = 10
UPLOAD_CHUNK_SIZE = 1024 * 1024
# Multipart headers and boundaries add bytes beyond the file payloads.
MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024


def max_document_size_bytes() -> int:
    try:
        size = int(get_setting("MAX_DOCUMENT_SIZE", str(DEFAULT_MAX_FILE_SIZE_BYTES)))
    except ValueError:
        return DEFAULT_MAX_FILE_SIZE_BYTES
    return size if size > 0 else DEFAULT_MAX_FILE_SIZE_BYTES


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if value:
            self._parts.append(value)

    def get_text(self) -> str:
        return "\n".join(self._parts)


class RagUploadService:
    @property
    def accepted_formats(self) -> list[str]:
        return sorted(SUPPORTED_EXTENSIONS)

    async def ingest_files(
        self,
        files: Sequence[UploadFile],
        graph_id: str,
    ) -> list[UploadedDocumentResult]:
        max_file_size = max_document_size_bytes()

        try:
            max_pages = int(get_setting("MAX_DOCUMENT_PAGES", "50"))
        except ValueError:
            max_pages = 50
        if max_pages < 1:
            max_pages = 50

        results: list[UploadedDocumentResult] = []

        for file in files:
            # Release the previous document's buffers before reading the next file.
            raw = b""
            text = ""
            prepared = ""
            filename = file.filename or "unnamed"
            extension = Path(filename).suffix.lower()

            if extension not in SUPPORTED_EXTENSIONS:
                results.append(
                    UploadedDocumentResult(
                        filename=filename,
                        status="skipped",
                        message=(
                            "Unsupported format. Use one of: "
                            f"{', '.join(self.accepted_formats)}"
                        ),
                    )
                )
                continue

            declared_size = file.headers.get("content-length")
            try:
                too_large = (
                    declared_size is not None and int(declared_size) > max_file_size
                )
            except ValueError:
                too_large = False

            raw_buffer = bytearray()
            while not too_large:
                chunk = await file.read(
                    min(UPLOAD_CHUNK_SIZE, max_file_size - len(raw_buffer) + 1)
                )
                if not chunk:
                    break
                if len(raw_buffer) + len(chunk) > max_file_size:
                    too_large = True
                    break
                raw_buffer.extend(chunk)

            if too_large:
                del raw_buffer
                results.append(
                    UploadedDocumentResult(
                        filename=filename,
                        status="skipped",
                        message=(
                            f"File is too large (max {max_file_size // (1024 * 1024)} MB)"
                        ),
                    )
                )
                continue

            raw = bytes(raw_buffer)
            del raw_buffer

            try:
                text = await self._extract_text(
                    raw=raw, extension=extension, max_pages=max_pages
                )
            except ValueError as exc:
                results.append(
                    UploadedDocumentResult(
                        filename=filename,
                        status="skipped",
                        message=str(exc),
                    )
                )
                continue

            prepared = get_abbrev_expander().expand(text.strip())
            if not prepared:
                results.append(
                    UploadedDocumentResult(
                        filename=filename,
                        status="skipped",
                        message="File is empty after text extraction",
                    )
                )
                continue

            if extension == ".pdf":
                generated_title = await generate_title_from_text(prepared)
                if generated_title and generated_title != "Без названия":
                    if not prepared.startswith(f"# {generated_title}"):
                        prepared = f"# {generated_title}\n\n{prepared}"
                    # Обновляем имя, чтобы отправить клиенту
                    # нормальное название и оригинальное в скобках
                    filename = f"{generated_title} ({filename})"

            source_id, file_paths_str = extract_sources(prepared, fallback=filename)
            source_url = first_http_url(file_paths_str)
            async with AsyncSessionLocal() as session:
                await ContentProposalService(session).create(
                    source="parser",
                    graph_id=graph_id,
                    title=source_id,
                    source_url=source_url,
                    text=prepared,
                )

            results.append(
                UploadedDocumentResult(
                    filename=filename,
                    status="queued",
                    message="Waiting for editor review",
                    chars=len(prepared),
                )
            )

        return results

    async def _extract_text(self, raw: bytes, extension: str, max_pages: int = 50) -> str:
        if extension == ".pdf":
            images = pdf_to_base64_images(raw, max_pages=max_pages)
            if not images:
                raise ValueError("Could not extract images from PDF")

            text = await parse_images_with_llm(images)
            if not text:
                raise ValueError("Could not extract text from PDF")
            return text

        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Only UTF-8 encoded files are supported") from exc

        if extension in {".txt", ".md", ".markdown"}:
            return text

        if extension == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON file") from exc
            return json.dumps(data, ensure_ascii=False, indent=2)

        if extension == ".csv":
            reader = csv.reader(io.StringIO(text))
            rows = [" | ".join(row) for row in reader]
            return "\n".join(rows)

        if extension in {".html", ".htm"}:
            parser = _HTMLTextExtractor()
            parser.feed(text)
            parser.close()
            return parser.get_text()

        raise ValueError("Unsupported file extension")
