import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.rag import router
from api.services.materials import page_materials
from db.postgres.db import get_async_session


def test_materials_pages_use_stable_global_order_and_filter_before_slicing():
    docs = [
        {
            "id": str(i),
            "title": f"Материал {i}",
            "created_at": "2026-11-11",
            "lightrag_status": "processed" if i % 2 else "failed",
        }
        for i in range(25)
    ]
    first = page_materials(docs, limit=10, state="ready")
    second = page_materials(docs, limit=10, offset=10, state="ready")
    assert first["total"] == 12 and first["total_all"] == 25
    assert len(first["documents"]) == 10 and len(second["documents"]) == 2
    assert not {d["id"] for d in first["documents"]} & {
        d["id"] for d in second["documents"]
    }
    assert first == page_materials(list(reversed(docs)), limit=10, state="ready")
    result = page_materials(docs, limit=10, query=" МАТЕРИАЛ 24 ")
    assert result["total"] == 1 and result["documents"][0]["id"] == "24"


def test_pending_flag_is_global_and_import_state_overrides_processed():
    docs = [
        {"id": "a", "lightrag_status": "processed"},
        {"id": "b", "lightrag_status": "processed", "import_state": "processing"},
    ]
    result = page_materials(docs, limit=1, query="a", state="ready")
    assert result["has_pending"] and result["total"] == 1
    assert page_materials(docs, state="attention")["documents"][0]["id"] == "b"
    assert page_materials([], limit=20)["total_all"] == 0


@pytest.mark.parametrize(
    "query",
    ["limit=0", "limit=101", "offset=-1", "state=invalid", "q=" + "x" * 501],
)
def test_materials_page_rejects_invalid_query_parameters(query):
    app = FastAPI()
    app.include_router(router)

    async def no_database():
        yield None

    app.dependency_overrides[get_async_session] = no_database
    with TestClient(app) as client:
        assert client.get(f"/rag/docs?{query}").status_code == 422
