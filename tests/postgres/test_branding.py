from io import BytesIO

from cryptography.fernet import Fernet
from httpx import ASGITransport, AsyncClient
from PIL import Image

from api.auth.dependencies import get_current_admin
from api.main import app
from db.postgres.db import get_async_session
from db.postgres.models import Admin, AdminRole
from db.postgres.services.instance_settings import InstanceSettingsService


def picture(color="red", format="PNG"):
    source = BytesIO()
    Image.new("RGB", (120, 40), color).save(source, format=format)
    return source.getvalue()


async def test_atomic_brand_lifecycle_and_public_whitelist(
    session, monkeypatch, tmp_path
):
    monkeypatch.setenv("BRANDING_MEDIA_DIR", str(tmp_path))
    monkeypatch.setenv("SETTINGS_ENCRYPTION_KEY", Fernet.generate_key().decode())
    service = InstanceSettingsService(session)
    await service.write(
        {
            "UNIVERSITY_PROFILE": "custom",
            "ADMISSIONS_EMAIL": "private@example.org",
            "OPENAI_API_KEY": "private-key",
            "ASSISTANT_NAME": "unchanged",
        },
        0,
    )
    admin = Admin(username="brand", password_hash="unused", role=AdminRole.superadmin)

    async def db():
        yield session

    app.dependency_overrides[get_async_session] = db
    app.dependency_overrides[get_current_admin] = lambda: admin
    try:
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://test"
        ) as client:

            async def upload(revision, content, changes="{}"):
                return await client.post(
                    "/api/v1/settings/instance/branding",
                    data={"revision": str(revision), "changes": changes},
                    files={"logo": ("../../logo.svg", content, "image/svg+xml")},
                )

            saved = await upload(
                1,
                picture(),
                '{"UNIVERSITY_ACCENT_COLOR":"#006633","UNIVERSITY_SHORT_NAME":"Север"}',
            )
            assert saved.status_code == 200, saved.text
            brand = (await client.get("/api/v1/branding")).json()
            assert set(brand) == {
                "revision",
                "short_name",
                "full_name",
                "logo_url",
                "favicon_url",
                "accent_color",
            }
            assert brand["short_name"] == "Север"
            assert brand["accent_color"] == "#006633"
            assert len(list(tmp_path.iterdir())) == 2
            old = brand["logo_url"]
            for content in (b"<svg/>", b"x" * (2 * 1024 * 1024 + 1)):
                assert (await upload(2, content)).status_code == 400
                assert (await client.get("/api/v1/branding")).json() == brand
            assert (await upload(1, picture("blue"))).status_code == 409
            assert len(list(tmp_path.iterdir())) == 2
            invalid = await upload(2, picture(), '{"UNIVERSITY_ACCENT_COLOR":"garbage"}')
            assert invalid.status_code == 400
            assert len(list(tmp_path.iterdir())) == 2
            icon = await client.get(brand["favicon_url"])
            assert icon.headers["content-type"] == "image/png"
            assert "immutable" in icon.headers["cache-control"]
            assert Image.open(BytesIO(icon.content)).size == (64, 64)
            assert (await upload(2, picture("blue", "WEBP"))).status_code == 200
            replacement = (await client.get("/api/v1/branding")).json()
            assert replacement["favicon_url"] != brand["favicon_url"]
            assert (await client.get(old)).status_code == 200
            removed = await client.post(
                "/api/v1/settings/instance/branding",
                data={"revision": "3", "remove_logo": "true"},
            )
            assert removed.status_code == 200
            after = (await client.get("/api/v1/branding")).json()
            assert after["logo_url"] == after["favicon_url"] == ""
            values = (await service.read()).values
            assert values["ADMISSIONS_EMAIL"] == "private@example.org"
            assert values["ASSISTANT_NAME"] == "unchanged"
            assert values["OPENAI_API_KEY"] == "private-key"
            admin.role = AdminRole.viewer
            assert (await upload(4, picture())).status_code == 403
            app.dependency_overrides.pop(get_current_admin)
            assert (await upload(4, picture())).status_code == 401
            assert (await client.get("/api/v1/branding")).status_code == 200
            assert (
                await client.get("/api/v1/branding/media/anything.svg")
            ).status_code == 404
    finally:
        app.dependency_overrides.clear()


async def test_concurrent_uploads_commit_only_one_revision(
    test_engine, monkeypatch, tmp_path
):
    import asyncio

    from sqlalchemy.ext.asyncio import async_sessionmaker

    monkeypatch.setenv("BRANDING_MEDIA_DIR", str(tmp_path))
    maker = async_sessionmaker(test_engine, expire_on_commit=False)

    async def db():
        async with maker() as connection:
            yield connection

    admin = Admin(username="brand", password_hash="unused", role=AdminRole.superadmin)
    app.dependency_overrides[get_async_session] = db
    app.dependency_overrides[get_current_admin] = lambda: admin
    try:
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://test"
        ) as client:
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/api/v1/settings/instance/branding",
                        data={"revision": "0"},
                        files={"logo": ("logo.png", picture(color), "image/png")},
                    )
                    for color in ("red", "blue")
                ]
            )
            assert sorted(response.status_code for response in responses) == [200, 409]
            assert len(list(tmp_path.iterdir())) == 2
            brand = (await client.get("/api/v1/branding")).json()
            assert brand["revision"] == 1
            assert (await client.get(brand["logo_url"])).status_code == 200
            assert (await client.get(brand["favicon_url"])).status_code == 200
    finally:
        app.dependency_overrides.clear()
