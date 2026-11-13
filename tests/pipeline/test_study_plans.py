import json
import re
from types import SimpleNamespace

import fitz
import httpx
import pytest
from fakeredis.aioredis import FakeRedis

from abbrev.expander import AbbrevExpander
from parser import url as url_parser
from pipeline import llm_client, study_plans

_PLAN_URL = (
    "https://www.nsu.ru/n/information-technologies-department/education_fit/"
    "programs/OOP/09-03-01/piikn/UP_09.03.01_2026_PIiKN.pdf"
)
_OCR_QUOTE = "Дискретная математика — 3-й семестр, обязательная дисциплина."


def _curriculum_pdf(*, scan: bool = False) -> bytes:
    font = fitz.Font("cjk")

    def write(page: fitz.Page, cells: list[tuple[int, int, str]]) -> None:
        writer = fitz.TextWriter(page.rect)
        for x, y, content in cells:
            writer.append((x, y), content, font=font, fontsize=9)
        writer.write_text(page)

    with fitz.open() as document:
        write(
            document.new_page(),
            [
                (30, 40, "Учебный план"),
                (30, 55, "09.03.01 Программная инженерия и компьютерные науки"),
                (30, 70, "Год начала подготовки 2026"),
            ],
        )
        if scan:
            with fitz.open() as scan_document:
                page = scan_document.new_page()
                write(page, [(30, 60, _OCR_QUOTE)])
                image = page.get_pixmap().tobytes("png")
                scan_page = document.new_page()
                scan_page.insert_image(scan_page.rect, stream=image)
            return document.tobytes()

        write(
            document.new_page(),
            [
                (370, 40, "По семестрам"),
                (490, 40, "Количество часов"),
                (100, 55, "Название дисциплины"),
                (300, 65, "Зачёт"),
                (400, 65, "Экзамен"),
                (30, 95, "Б1.Б"),
                (100, 95, "Обязательная часть"),
                (30, 110, "Б1.Б.11"),
                (100, 110, "Дискретная математика"),
                (410, 110, "3"),
                (500, 110, "8"),
            ],
        )
        return document.tobytes()


class _NoFaq:
    async def match_async(self, user_question: str) -> None:
        return None


@pytest.fixture
async def plan_pipeline(monkeypatch, request):
    payload = _curriculum_pdf(scan=getattr(request, "param", False))
    async with FakeRedis(decode_responses=True) as redis:
        history = object.__new__(llm_client.RedisClient)
        history.client = redis
        expander = AbbrevExpander()
        expander.load_items([])
        fetched = []

        async def get_history_client():
            return history

        async def faq_threshold() -> float:
            return 0.99

        async def fetch(url: str, **kwargs) -> httpx.Response:
            assert url == _PLAN_URL
            fetched.append(url)
            return httpx.Response(
                200,
                request=httpx.Request("GET", url),
                headers={"Content-Type": "application/pdf"},
                content=payload,
            )

        async def forbid_rag(*args, **kwargs):
            raise AssertionError("Generic retrieval cannot verify curriculum facts")

        async def load_crag_config():
            return SimpleNamespace(enabled=False)

        def forbid_model():
            raise AssertionError("Native PDF should be sufficient to verify this row")

        monkeypatch.setenv("UNIVERSITY_PROFILE", "nsu")
        monkeypatch.setattr(study_plans, "_cached_plans", {})
        monkeypatch.setattr(url_parser, "fetch_url_async", fetch)
        monkeypatch.setattr(llm_client, "get_redis_client", get_history_client)
        monkeypatch.setattr(llm_client, "get_abbrev_expander", lambda: expander)
        monkeypatch.setattr(llm_client, "get_faq_matcher", _NoFaq)
        monkeypatch.setattr(llm_client, "load_faq_threshold", faq_threshold)
        monkeypatch.setattr(llm_client, "load_crag_config", load_crag_config)
        monkeypatch.setattr(llm_client, "get_llm_provider", forbid_model)
        monkeypatch.setattr(llm_client, "query_graph_with_sources", forbid_rag)
        monkeypatch.setattr(llm_client, "query_graph_with_crag", forbid_rag)
        monkeypatch.setattr(llm_client, "_spawn_bg", lambda coro: coro.close())
        yield fetched


@pytest.mark.parametrize("cohort", [2026, 2025])
async def test_bot_answers_with_confirmed_pdf_page_and_cohort(plan_pipeline, cohort):
    question = (
        f"В каком семестре у набора {cohort} ПИиКН дискретная математика "
        "и обязательна ли она?"
    )
    if cohort != 2026:
        question += f" Сверь с этим PDF: {_PLAN_URL}"
    answer = await llm_client.ask_local_llm(question, "curriculum-test")

    assert plan_pipeline == [_PLAN_URL]
    assert "дискретная математика" in answer.text.casefold()
    assert re.search(r"3(?:-\w+)?\s+семестр", answer.text.casefold())
    assert "обязательн" in answer.text.casefold()
    assert "2026" in answer.text
    assert not re.search(r"8(?:-\w+)?\s+семестр", answer.text.casefold())
    assert [source["url"] for source in answer.sources] == [f"{_PLAN_URL}#page=2"]
    other_cohort = re.search(r"друг(?:ого|ой) набор(?:а)?", answer.text.casefold())
    assert bool(other_cohort) == (cohort != 2026)
    if cohort != 2026:
        assert "2025" in answer.text


@pytest.mark.parametrize("plan_pipeline", [True], indirect=True)
@pytest.mark.parametrize("evidence", ["verified", "invented", "wrong-page", "joined"])
async def test_bot_uses_ocr_only_with_verified_page_evidence(
    plan_pipeline, evidence, monkeypatch
) -> None:
    async def ocr(images: list[str]) -> str:
        assert len(images) == 1
        if evidence == "joined":
            return _OCR_QUOTE.replace(" — ", "\n— ", 1)
        return _OCR_QUOTE

    quote = _OCR_QUOTE.replace("3-й", "9-й") if evidence == "invented" else _OCR_QUOTE
    page = 1 if evidence == "wrong-page" else 2

    class Provider:
        async def generate(self, messages, profile=None) -> str:
            assert any("Дискретная математика" in str(m.content) for m in messages)
            return json.dumps(
                [{"page": page, "name": "Дискретная математика", "quote": quote}],
                ensure_ascii=False,
            )

    monkeypatch.setattr(url_parser, "parse_images_with_llm", ocr)
    monkeypatch.setattr(study_plans, "get_llm_provider", Provider)
    question = (
        "В каком семестре у набора 2026 ПИиКН дискретная математика "
        f"и обязательна ли она? Сверь с этим PDF: {_PLAN_URL}"
    )
    answer = await llm_client.ask_local_llm(question, "curriculum-ocr-test")

    assert plan_pipeline == [_PLAN_URL]
    if evidence == "verified":
        assert re.search(r"3(?:-\w+)?\s+семестр", answer.text.casefold())
        assert "обязательн" in answer.text.casefold()
        assert [source["url"] for source in answer.sources] == [f"{_PLAN_URL}#page=2"]
    else:
        assert "подтвердить" in answer.text.casefold()
        assert not re.search(r"9(?:-\w+)?\s+семестр", answer.text.casefold())
        assert not answer.sources
