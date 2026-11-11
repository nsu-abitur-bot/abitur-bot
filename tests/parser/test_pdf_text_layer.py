import fitz
import pytest

from parser.pdf import extract_pdf_text


def document(pages=1):
    with fitz.open() as pdf:
        for _ in range(pages):
            page = pdf.new_page()
            page.insert_text((50, 50), "Admission information")
        return pdf.tobytes()


def test_pdf_text_is_available_without_a_vision_provider():
    assert "Admission information" in extract_pdf_text(document())


def test_text_extraction_respects_the_same_page_limit():
    with pytest.raises(ValueError, match="maximum allowed pages"):
        extract_pdf_text(document(2), max_pages=1)
