import base64
import logging
from dataclasses import dataclass
from typing import List

import fitz

from parser.study_plan_table import PdfDiscipline, read_study_plan_table

logger = logging.getLogger(__name__)


def pdf_to_base64_images(pdf_bytes: bytes, max_pages: int = 50) -> List[str]:
    """Конвертирует страницы PDF во множество base64-строк формата JPEG"""
    images = []
    # fitz.open(stream) -> filetype должен быть "pdf"
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if len(doc) > max_pages:
            raise ValueError(f"PDF exceeds maximum allowed pages ({max_pages})")
        for page in doc:
            # Масштаб 2.0 дает ~150-300 DPI, что достаточно для качественного OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img_bytes = pix.tobytes("jpeg")
            images.append(base64.b64encode(img_bytes).decode("utf-8"))
    return images


@dataclass(frozen=True)
class PdfPage:
    number: int
    text: str
    disciplines: tuple[PdfDiscipline, ...] = ()


def read_pdf_pages(pdf_bytes: bytes, max_pages: int) -> list[tuple[PdfPage, str | None]]:
    """Return physical pages with an OCR image only where the text layer is absent."""
    pages: list[tuple[PdfPage, str | None]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if not doc.is_pdf:
            raise ValueError("Expected a PDF document")
        if len(doc) > max_pages:
            raise ValueError(f"PDF exceeds maximum allowed pages ({max_pages})")
        for index in range(len(doc)):
            page = doc.load_page(index)
            number = index + 1
            text_page = page.get_textpage()
            text = text_page.extractText(sort=True).strip()
            letter_count = sum(char.isalpha() for char in text)
            images = page.get_image_info() if letter_count < 80 else []
            scanned = bool(images) and (
                not letter_count
                or max(fitz.Rect(image["bbox"]).get_area() for image in images)
                >= page.rect.get_area() * 0.45
            )
            if letter_count and not scanned:
                disciplines = read_study_plan_table(text_page.extractWORDS())
                pages.append((PdfPage(number, text, disciplines), None))
            elif scanned:
                # Wide plans need enough resolution for their smallest columns.
                # Bound the pixel count even for PDFs with oversized page boxes.
                scale = min(2.0, 3200 / max(page.rect.width, page.rect.height))
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                image = base64.b64encode(pixmap.tobytes("jpeg")).decode("ascii")
                pages.append((PdfPage(number, ""), image))
            else:
                pages.append((PdfPage(number, text), None))
    return pages
