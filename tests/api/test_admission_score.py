"""Тесты API проходных баллов (TestClient + фейк-сервис, парсер замокан)."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes.admission_score import get_admission_score_service
from db.postgres.services.admission_score import ScoreRow


class FakeAdmissionScoreService:
    """Минимальная имитация AdmissionScoreService в памяти."""

    def __init__(self):
        # Известные (факультет, направление) → program_id.
        self.known = {
            ("Факультет информационных технологий", "Программная инженерия"): "prog-1",
        }
        self.upserted: list[ScoreRow] = []
        self.query_result: list[dict] = []
        self.years: list[int] = [2024, 2023]

    async def resolve_program_id(self, faculty_name, program_name, level="bachelor"):
        return self.known.get((faculty_name, program_name))

    async def upsert_from_rows(self, rows):
        self.upserted = list(rows)
        return {"created": len(self.upserted), "updated": 0, "skipped": 0}

    async def query_scores(self, *, faculty=None, program=None, year=None, form=None):
        return self.query_result

    async def get_available_years(self):
        return self.years


@pytest.fixture
def service() -> FakeAdmissionScoreService:
    return FakeAdmissionScoreService()


@pytest.fixture
def client(service: FakeAdmissionScoreService):
    app.dependency_overrides[get_admission_score_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_admission_score_service, None)


def _parsed_rows():
    return [
        ScoreRow(
            faculty_name="Факультет информационных технологий",
            program_name="Программная инженерия",
            code="09.03.01",
            year=2024,
            form="budget",
            passing_score=246,
            average_score=260.9,
        ),
        ScoreRow(
            faculty_name="Неизвестный факультет",
            program_name="Неизвестное направление",
            code=None,
            year=2024,
            form="budget",
            passing_score=200,
            average_score=210.0,
        ),
    ]


def test_preview_counts_matched_and_unmatched(client: TestClient):
    async def fake_parse(url):
        return _parsed_rows()

    with patch("api.routes.admission_score.parse_scores", fake_parse):
        resp = client.post("/api/v1/admission-scores/preview", json={})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["rows"]) == 2
    summary = body["summary"]
    assert summary["total"] == 2
    assert summary["matched"] == 1
    assert summary["unmatched"] == 1
    assert summary["unmatched_samples"] == [
        {
            "faculty_name": "Неизвестный факультет",
            "program_name": "Неизвестное направление",
        }
    ]


def test_import_returns_stats(client: TestClient, service: FakeAdmissionScoreService):
    payload = {
        "rows": [
            {
                "faculty_name": "Факультет информационных технологий",
                "program_name": "Программная инженерия",
                "code": "09.03.01",
                "year": 2024,
                "form": "budget",
                "passing_score": 246,
                "average_score": 260.9,
                "level": "bachelor",
            }
        ]
    }
    resp = client.post("/api/v1/admission-scores/import", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"created": 1, "updated": 0, "skipped": 0}
    assert len(service.upserted) == 1
    assert service.upserted[0].passing_score == 246


def test_get_scores(client: TestClient, service: FakeAdmissionScoreService):
    service.query_result = [
        {
            "faculty_name": "Факультет информационных технологий",
            "program_name": "Программная инженерия",
            "code": "09.03.01",
            "year": 2024,
            "form": "budget",
            "passing_score": 246,
            "average_score": 260.9,
        }
    ]
    resp = client.get("/api/v1/admission-scores", params={"year": 2024})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["passing_score"] == 246


def test_get_years(client: TestClient):
    resp = client.get("/api/v1/admission-scores/years")
    assert resp.status_code == 200
    assert resp.json() == [2024, 2023]
