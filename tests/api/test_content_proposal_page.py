from fastapi.testclient import TestClient

from api.main import app


def test_review_page_is_available() -> None:
    response = TestClient(app).get("/review/proposals")
    assert response.status_code == 200
    assert "Предложения материалов" in response.text
    assert "Принять" in response.text
    assert "Отклонить" in response.text
