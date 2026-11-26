import json

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from config.content_gate import migration_marker


@pytest.mark.asyncio
async def test_model_admin_routes_require_authentication_even_during_migration(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(app, "dependency_overrides", {})
    monkeypatch.setenv("LIGHTRAG_WORKSPACE_BASE", str(tmp_path))
    migration_marker().write_text(json.dumps({"id": "pending"}))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        cases = [
            ("GET", "/api/v1/settings/instance/models", None),
            ("POST", "/api/v1/settings/instance/models", {"revision": 1, "changes": {}}),
            ("POST", "/api/v1/settings/instance/models/job/test/cancel", None),
            ("POST", "/api/v1/faq", {"question": "Q", "answer": "A", "aliases": []}),
        ]
        for method, url, data in cases:
            response = await client.request(method, url, json=data)
            assert response.status_code in {401, 403}
