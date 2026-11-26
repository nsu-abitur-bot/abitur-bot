import base64

import fitz
import pytest

from parser import url as url_parser


@pytest.mark.parametrize("text_stamp", [False, True])
async def test_mixed_pdf_keeps_physical_page_numbers_when_ocr_returns_no_text(
    monkeypatch, text_stamp: bool
) -> None:
    with fitz.open() as scan_source, fitz.open() as document:
        scan = scan_source.new_page(width=280, height=100)
        scan.insert_text((20, 30), "Scanned curriculum row")
        image = scan.get_pixmap().tobytes("png")

        first = document.new_page(width=280, height=100)
        first.insert_text((20, 30), "Curriculum for cohort 2026")
        second = document.new_page(width=280, height=100)
        second.insert_image(second.rect, stream=image)
        if text_stamp:
            second.insert_text((20, 95), "Approved")
        third = document.new_page(width=280, height=100)
        third.insert_image(fitz.Rect(0, 0, 140, 50), stream=image)
        fourth = document.new_page(width=280, height=100)
        fourth.insert_text((20, 30), "Final curriculum row")
        payload = document.tobytes()

    ocr_calls: list[bytes] = []

    async def ocr(images: list[str]) -> str:
        assert len(images) == 1
        ocr_calls.append(base64.b64decode(images[0], validate=True))
        return "Scanned curriculum row" if len(ocr_calls) == 1 else ""

    monkeypatch.setattr(url_parser, "parse_images_with_llm", ocr)

    pages = await url_parser.extract_pdf_pages(payload)

    assert [page.number for page in pages] == [1, 2, 3, 4]
    assert pages[0].text.strip() == "Curriculum for cohort 2026"
    assert "Scanned curriculum row" in pages[1].text
    assert not pages[2].text.strip()
    assert pages[3].text.strip() == "Final curriculum row"
    assert len(ocr_calls) == 2
    assert ocr_calls[0] != ocr_calls[1]
