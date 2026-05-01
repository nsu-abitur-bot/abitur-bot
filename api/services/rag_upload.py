import csv
import io
import json
from html.parser import HTMLParser
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import select

from abbrev.expander import get_abbrev_expander
from api.schemas.rag import UploadedDocumentResult
from db.postgres.db import AsyncSessionLocal
from db.postgres.models import Settings
from llm.preprocessor import generate_title_from_text
from llm.vision_parser import parse_images_with_llm
from parser.pdf import pdf_to_base64_images
from parser.utils import extract_sources
from rag.loader import add_texts_async

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

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # увеличен лимит до 50MB из-за PDF


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

    async def _get_settings(self) -> dict[str, str]:
        async with AsyncSessionLocal() as session:
            stmt = select(Settings)
            result = await session.execute(stmt)
            return {s.key: s.value for s in result.scalars()}

    async def ingest_files(
        self,
        files: list[UploadFile],
        graph_id: str,
    ) -> list[UploadedDocumentResult]:
        settings = await self._get_settings()
        try:
            max_file_size = int(settings.get("max_document_size", 50 * 1024 * 1024))
        except ValueError:
            max_file_size = 50 * 1024 * 1024

        try:
            max_pages = int(settings.get("max_document_pages", 50))
        except ValueError:
            max_pages = 50

        results: list[UploadedDocumentResult] = []

        for file in files:
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

            raw = await file.read()
            if len(raw) > max_file_size:
                results.append(
                    UploadedDocumentResult(
                        filename=filename,
                        status="skipped",
                        message=(
                            "File is too large "
                            f"(max {max_file_size // (1024 * 1024)} MB)"
                        ),
                    )
                )
                continue

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
                    # Обновляем имя, чтобы отправить клиенту нормальное название и оригинальное в скобках
                    filename = f"{generated_title} ({filename})"

            source_id, file_paths_str = extract_sources(prepared, fallback=filename)

            # Передаем подготовленный текст и извлеченные ID/источники в RAG
            saved_count = await add_texts_async(
                [prepared],
                graph_id=graph_id,
                source_ids=[source_id],
                file_paths=[file_paths_str],
            )

            if saved_count == 0:
                results.append(
                    UploadedDocumentResult(
                        filename=filename,
                        status="skipped",
                        message="Failed to save document to RAG",
                    )
                )
                continue

            results.append(
                UploadedDocumentResult(
                    filename=filename,
                    status="indexed",
                    message="Document indexed in RAG",
                    chars=len(prepared),
                )
            )

        return results

    async def _extract_text(
        self, raw: bytes, extension: str, max_pages: int = 50
    ) -> str:
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
