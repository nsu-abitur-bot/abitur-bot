import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes.faq import get_faq_service
from api.schemas.faq import FaqItem

SAMPLE_FAQ_ITEMS = [
    FaqItem(
        id="faq-1",
        question="TEST_QUESTION_1",
        aliases=["ALIAS_1", "ALIAS_2"],
        answer="TEST_ANSWER_1",
    ),
    FaqItem(
        id="faq-2",
        question="TEST_QUESTION_2",
        aliases=[],
        answer="TEST_ANSWER_2",
    ),
]


class FakeFaqService:
    def __init__(self, items: list[FaqItem]):
        self.items = [item.model_copy(deep=True) for item in items]
        self._next_id = len(items) + 1

    async def get_all(self) -> list[FaqItem]:
        return [item.model_copy(deep=True) for item in self.items]

    async def create(self, item: FaqItem) -> FaqItem:
        created = item.model_copy(update={"id": f"faq-{self._next_id}"})
        self._next_id += 1
        self.items.append(created)
        return created.model_copy(deep=True)

    async def update(self, item_id: str, item: FaqItem) -> FaqItem:
        for index, existing in enumerate(self.items):
            if existing.id == item_id:
                updated = item.model_copy(update={"id": item_id})
                self.items[index] = updated
                return updated.model_copy(deep=True)
        raise KeyError(item_id)

    async def delete(self, item_id: str) -> None:
        for index, item in enumerate(self.items):
            if item.id == item_id:
                del self.items[index]
                return
        raise KeyError(item_id)


@pytest.fixture
def faq_service() -> FakeFaqService:
    return FakeFaqService(SAMPLE_FAQ_ITEMS)


@pytest.fixture
def client(faq_service: FakeFaqService):
    app.dependency_overrides[get_faq_service] = lambda: faq_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_faq_service, None)


def test_get_faq_list(client: TestClient):
    response = client.get("/api/v1/faq")

    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 2
    assert data["items"][0]["question"] == "TEST_QUESTION_1"
    assert "ALIAS_1" in data["items"][0]["aliases"]


def test_create_faq_item(client: TestClient):
    new_item = {
        "question": "NEW_QUESTION",
        "aliases": ["NEW_ALIAS"],
        "answer": "NEW_ANSWER",
    }

    response = client.post("/api/v1/faq", json=new_item)

    assert response.status_code == 201
    data = response.json()
    assert data["id"] == "faq-3"
    assert data["question"] == "NEW_QUESTION"

    get_response = client.get("/api/v1/faq")
    items = get_response.json()["items"]
    assert len(items) == 3
    assert items[-1]["question"] == "NEW_QUESTION"


def test_update_faq_item(client: TestClient):
    update_data = {
        "question": "UPDATED_QUESTION_1",
        "aliases": ["UPDATED_ALIAS"],
        "answer": "UPDATED_ANSWER_1",
    }

    response = client.put("/api/v1/faq/faq-1", json=update_data)

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "faq-1"
    assert data["question"] == "UPDATED_QUESTION_1"

    get_response = client.get("/api/v1/faq")
    items = get_response.json()["items"]
    assert items[0]["question"] == "UPDATED_QUESTION_1"


def test_update_faq_item_not_found(client: TestClient):
    update_data = {
        "question": "Q",
        "aliases": [],
        "answer": "A",
    }

    response = client.put("/api/v1/faq/missing", json=update_data)

    assert response.status_code == 404


def test_delete_faq_item(client: TestClient):
    response = client.delete("/api/v1/faq/faq-1")

    assert response.status_code == 204

    get_response = client.get("/api/v1/faq")
    items = get_response.json()["items"]
    assert len(items) == 1
    assert items[0]["question"] == "TEST_QUESTION_2"


def test_delete_faq_item_not_found(client: TestClient):
    response = client.delete("/api/v1/faq/missing")

    assert response.status_code == 404
