import base64
import logging
from typing import List

import fitz

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
