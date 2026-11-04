from datetime import datetime

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from api.routes.faq import get_faq_service, router
from api.schemas.faq import FaqItem
from api.services.faq import FaqService


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["confirmed_by", "confirmed_at"])
async def test_invalid_merged_update_is_atomic(session: AsyncSession, field: str) -> None:
    service = FaqService(session)
    created = await service.create(
        FaqItem(
            question="q",
            answer="a",
            verification_status="confirmed",
            confirmed_by="Reviewer",
            confirmed_at=datetime(2026, 11, 4, 12),
        )
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_faq_service] = lambda: service
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.put(
            f"/faq/{created.id}",
            json={
                "question": "changed",
                "answer": "changed",
                field: None,
            },
        )
        assert response.status_code == 422
        async with AsyncSession(bind=session.bind) as fresh_session:
            assert await FaqService(fresh_session).get_all() == [created]
        assert (await client.get("/faq")).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "date",
    [
        "2026-11-04T12:00:00Z",
        "2026-11-04T15:00:00+03:00",
        "2026-11-04T12:00:00",
    ],
)
@pytest.mark.parametrize("status", ["draft", "confirmed"])
async def test_dates_round_trip(session: AsyncSession, date: str, status: str) -> None:
    service = FaqService(session)
    payload = dict(
        question="q",
        answer="a",
        verification_status=status,
        confirmed_by="Reviewer",
        confirmed_at=date,
    )
    created = await service.create(FaqItem.model_validate(payload))
    assert created.confirmed_at == datetime(2026, 11, 4, 12)
    updated = await service.update(created.id or "", FaqItem.model_validate(payload))
    assert updated.confirmed_at == datetime(2026, 11, 4, 12)


@pytest.mark.parametrize(
    "origin",
    [
        {"confirmed_by": " \t "},
        {"source_document": " ", "source_location": "\n"},
        {"source_document": "Rules", "source_location": " "},
    ],
)
def test_whitespace_is_not_confirmation_origin(origin: dict) -> None:
    with pytest.raises(ValidationError):
        FaqItem(
            question="q",
            answer="a",
            verification_status="confirmed",
            confirmed_at=datetime(2026, 11, 4, 12),
            **origin,
        )


def test_draft_provenance_normalization() -> None:
    item = FaqItem(
        question="q",
        answer="a",
        confirmed_by="  Reviewer  ",
        source_document=" ",
        source_location="  p. 1  ",
    )
    assert item.confirmed_by == "Reviewer"
    assert item.source_document is None
    assert item.source_location == "p. 1"


@pytest.mark.asyncio
async def test_explicit_null_and_omitted_draft_metadata(session: AsyncSession) -> None:
    service = FaqService(session)
    created = await service.create(
        FaqItem(
            question="q",
            answer="a",
            confirmed_by="Reviewer",
            source_document="Rules",
            source_location="p. 1",
            confirmed_at=datetime(2026, 11, 4, 12),
        )
    )
    updated = await service.update(
        created.id or "",
        FaqItem(
            question="q",
            answer="a",
            confirmed_by=None,
        ),
    )
    assert updated.confirmed_by is None
    assert updated.source_document == "Rules"
    assert updated.source_location == "p. 1"
    assert updated.confirmed_at == created.confirmed_at
    cleared = await service.update(
        created.id or "",
        FaqItem(
            question="q",
            answer="a",
            source_document=" ",
            confirmed_at=None,
        ),
    )
    assert cleared.source_document is None
    assert cleared.source_location == "p. 1"
    assert cleared.confirmed_at is None
