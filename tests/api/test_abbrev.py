import io

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes.abbrev import get_abbrev_service
from api.schemas.abbrev import AbbrevItem

SAMPLE_ABBREV_ITEMS = [
    AbbrevItem(id="abbrev-1", short="НГУ", full="Новосибирский гос. университет"),
    AbbrevItem(id="abbrev-2", short="ФИТ", full="Факультет информационных технологий"),
]


class FakeAbbrevService:
    def __init__(self, items: list[AbbrevItem]):
        self.items = [item.model_copy(deep=True) for item in items]
        self._next_id = len(items) + 1

    async def get_all(self) -> list[AbbrevItem]:
        return [item.model_copy(deep=True) for item in self.items]

    async def import_many(self, items: list[AbbrevItem]) -> list[AbbrevItem]:
        by_short = {item.short: item for item in self.items}
        for incoming in items:
            if incoming.short in by_short:
                by_short[incoming.short].full = incoming.full
            else:
                created = incoming.model_copy(update={"id": f"abbrev-{self._next_id}"})
                self._next_id += 1
                self.items.append(created)
                by_short[created.short] = created
        return await self.get_all()


@pytest.fixture
def abbrev_service() -> FakeAbbrevService:
    return FakeAbbrevService(SAMPLE_ABBREV_ITEMS)


@pytest.fixture
def client(abbrev_service: FakeAbbrevService):
    app.dependency_overrides[get_abbrev_service] = lambda: abbrev_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_abbrev_service, None)


def _csv_upload(content: str, filename: str = "abbrev.csv"):
    return {"file": (filename, io.BytesIO(content.encode("utf-8")), "text/csv")}


def test_upload_abbrev_csv_success(client: TestClient):
    csv_content = (
        "Аббревиатура,Расшифровка\r\n"
        "ММФ,Механико-математический факультет\r\n"
        "ФЕН,Факультет естественных наук\r\n"
    )

    response = client.post("/api/v1/abbrev/upload", files=_csv_upload(csv_content))

    assert response.status_code == 201
    shorts = {item["short"] for item in response.json()["items"]}
    assert "ММФ" in shorts
    assert "ФЕН" in shorts
    assert len(shorts) == 4


def test_upload_abbrev_csv_upserts_existing(client: TestClient):
    csv_content = "Аббревиатура,Расшифровка\r\nНГУ,Новое значение НГУ\r\n"

    response = client.post("/api/v1/abbrev/upload", files=_csv_upload(csv_content))

    assert response.status_code == 201
    items = response.json()["items"]
    ngu = next(item for item in items if item["short"] == "НГУ")
    assert ngu["full"] == "Новое значение НГУ"
    # запись не дублируется
    assert sum(1 for item in items if item["short"] == "НГУ") == 1


def test_upload_abbrev_csv_alternative_column_names(client: TestClient):
    csv_content = "Short,Full\r\nГГФ,Геолого-геофизический факультет\r\n"

    response = client.post("/api/v1/abbrev/upload", files=_csv_upload(csv_content))

    assert response.status_code == 201
    shorts = {item["short"] for item in response.json()["items"]}
    assert "ГГФ" in shorts


def test_upload_abbrev_csv_without_header(client: TestClient):
    # Файл без строки заголовка — первая строка не должна теряться.
    csv_content = "ГГФ,Геолого-геофизический факультет\r\nФФ,Физический факультет\r\n"

    response = client.post("/api/v1/abbrev/upload", files=_csv_upload(csv_content))

    assert response.status_code == 201
    shorts = {item["short"] for item in response.json()["items"]}
    assert "ГГФ" in shorts
    assert "ФФ" in shorts


def test_upload_abbrev_rejects_non_csv(client: TestClient):
    response = client.post(
        "/api/v1/abbrev/upload",
        files=_csv_upload("НГУ;Расшифровка", filename="data.txt"),
    )

    assert response.status_code == 400


def test_upload_abbrev_rejects_empty_content(client: TestClient):
    csv_content = "Аббревиатура,Расшифровка\r\n"

    response = client.post("/api/v1/abbrev/upload", files=_csv_upload(csv_content))

    assert response.status_code == 400
