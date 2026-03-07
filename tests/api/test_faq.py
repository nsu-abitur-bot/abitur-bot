
import pytest
from fastapi.testclient import TestClient

from api.main import app

# Use TestClient for synchronous testing of FastAPI
client = TestClient(app)

# A sample FAQ data for testing
SAMPLE_FAQ_DATA = {
    "faq": [
        {
            "question": "TEST_QUESTION_1",
            "aliases": ["ALIAS_1", "ALIAS_2"],
            "answer": "TEST_ANSWER_1",
        },
        {
            "question": "TEST_QUESTION_2",
            "aliases": [],
            "answer": "TEST_ANSWER_2",
        },
    ]
}


@pytest.fixture
def temp_faq_file(tmp_path):
    """Fixture to create a temporary FAQ yaml file and override the service path."""
    import yaml
    
    faq_file = tmp_path / "test_faq_data.yaml"
    with open(faq_file, "w", encoding="utf-8") as f:
        yaml.dump(SAMPLE_FAQ_DATA, f, allow_unicode=True)
    
    # We will need to patch the service and matcher to use this file
    # We'll do this by overriding a dependency or variable in the actual test later
    return faq_file


def test_get_faq_list(temp_faq_file, monkeypatch):
    """Test retrieving the list of FAQs."""
    # We need to mock the path in the service when it's created.
    # Assuming `api.services.faq.FAQ_DATA_PATH` is used.
    monkeypatch.setattr("api.services.faq.FAQ_DATA_PATH", temp_faq_file)
    
    response = client.get("/api/v1/faq")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 2
    assert data["items"][0]["question"] == "TEST_QUESTION_1"
    assert "ALIAS_1" in data["items"][0]["aliases"]


def test_create_faq_item(temp_faq_file, monkeypatch):
    """Test creating a new FAQ item."""
    monkeypatch.setattr("api.services.faq.FAQ_DATA_PATH", temp_faq_file)
    
    new_item = {
        "question": "NEW_QUESTION",
        "aliases": ["NEW_ALIAS"],
        "answer": "NEW_ANSWER"
    }
    response = client.post("/api/v1/faq", json=new_item)
    assert response.status_code == 201
    
    # Verify it was added
    data = response.json()
    assert data["question"] == "NEW_QUESTION"
    
    # Verify via GET
    get_response = client.get("/api/v1/faq")
    items = get_response.json()["items"]
    assert len(items) == 3
    assert items[-1]["question"] == "NEW_QUESTION"


def test_update_faq_item(temp_faq_file, monkeypatch):
    """Test updating an existing FAQ item."""
    monkeypatch.setattr("api.services.faq.FAQ_DATA_PATH", temp_faq_file)
    
    update_data = {
        "question": "UPDATED_QUESTION_1",
        "aliases": ["UPDATED_ALIAS"],
        "answer": "UPDATED_ANSWER_1"
    }
    # Update the first item (index 0)
    response = client.put("/api/v1/faq/0", json=update_data)
    assert response.status_code == 200
    
    # Verify update
    data = response.json()
    assert data["question"] == "UPDATED_QUESTION_1"
    
    # Verify via GET
    get_response = client.get("/api/v1/faq")
    items = get_response.json()["items"]
    assert items[0]["question"] == "UPDATED_QUESTION_1"


def test_update_faq_item_not_found(temp_faq_file, monkeypatch):
    """Test updating a non-existent FAQ item."""
    monkeypatch.setattr("api.services.faq.FAQ_DATA_PATH", temp_faq_file)
    
    update_data = {
        "question": "Q",
        "aliases": [],
        "answer": "A"
    }
    response = client.put("/api/v1/faq/999", json=update_data)
    assert response.status_code == 404


def test_delete_faq_item(temp_faq_file, monkeypatch):
    """Test deleting an FAQ item."""
    monkeypatch.setattr("api.services.faq.FAQ_DATA_PATH", temp_faq_file)
    
    response = client.delete("/api/v1/faq/0")
    assert response.status_code == 204
    
    # Verify deletion
    get_response = client.get("/api/v1/faq")
    items = get_response.json()["items"]
    assert len(items) == 1
    assert items[0]["question"] == "TEST_QUESTION_2"


def test_delete_faq_item_not_found(temp_faq_file, monkeypatch):
    """Test deleting a non-existent FAQ item."""
    monkeypatch.setattr("api.services.faq.FAQ_DATA_PATH", temp_faq_file)
    
    response = client.delete("/api/v1/faq/999")
    assert response.status_code == 404
