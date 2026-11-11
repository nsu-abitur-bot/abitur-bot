from collections.abc import AsyncIterator

from fakeredis.aioredis import FakeRedis
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from api.auth.dependencies import get_current_admin
from api.main import app
from bootstrap import runtime_settings
from bot import core
from config import instance
from db.postgres.db import get_async_session
from db.postgres.models import Admin, AdminRole, Base
from db.redis import client as redis_client


async def test_welcome_settings_apply_to_both_channels_and_stay_on_their_stand(
    test_engine: AsyncEngine, monkeypatch
) -> None:
    other_engine = test_engine.execution_options(
        schema_translate_map={None: "welcome_other_stand"}
    )
    async with other_engine.begin() as connection:
        await connection.execute(text("CREATE SCHEMA welcome_other_stand"))
        await connection.run_sync(Base.metadata.tables["settings"].create)

    nsu = async_sessionmaker(test_engine, expire_on_commit=False)
    other = async_sessionmaker(other_engine, expire_on_commit=False)
    stand = nsu

    async def settings_session() -> AsyncIterator[AsyncSession]:
        async with stand() as session:
            yield session

    admin = Admin(username="welcome", password_hash="unused", role=AdminRole.superadmin)
    monkeypatch.setitem(app.dependency_overrides, get_current_admin, lambda: admin)
    monkeypatch.setitem(app.dependency_overrides, get_async_session, settings_session)
    # The API and the bot have separate in-memory settings in production.
    monkeypatch.setattr("api.routes.instance_settings.install", lambda *_: None)
    monkeypatch.setattr(core, "AsyncSessionLocal", nsu)
    monkeypatch.setattr(redis_client, "REDIS_URL", "redis://localhost")
    bot = core.BotCore()
    path = "/api/v1/settings/instance"
    key = "BOT_WELCOME_MESSAGE"
    nsu_text = "  Привет из НГУ!\n\n<b>Условия</b> & **сроки**\n"
    other_text = "Добро пожаловать в СибГИУ!\nЗадайте вопрос."

    async def refresh() -> None:
        monkeypatch.setattr(runtime_settings, "AsyncSessionLocal", stand)
        await runtime_settings.refresh_settings()

    async def check_replies(expected: str, *, custom: bool) -> None:
        for channel in ("telegram", "max"):
            reply = await bot.cmd_start(channel, "123", "456")
            if custom:
                assert reply.text == expected
                assert reply.parse_mode is None
            else:
                assert expected in reply.text
                assert reply.parse_mode == "HTML"

    try:
        async with (
            FakeRedis(decode_responses=True) as redis,
            AsyncClient(transport=ASGITransport(app), base_url="http://test") as http,
        ):
            store = redis_client.RedisClient()
            store.client = redis
            monkeypatch.setattr(redis_client, "_redis_client", store)
            await refresh()
            await check_replies("«НГУ»", custom=False)

            async def save(revision: int, changes: dict[str, str]) -> None:
                response = await http.put(
                    path, json={"revision": revision, "changes": changes}
                )
                assert response.status_code == 200, response.text
                assert instance.revision() == revision
                await refresh()
                assert instance.revision() == revision + 1

            await save(0, {key: nsu_text})
            await check_replies(nsu_text, custom=True)

            stand = other
            await refresh()
            await save(
                0,
                {
                    "UNIVERSITY_PROFILE": "custom",
                    "UNIVERSITY_SHORT_NAME": "СибГИУ",
                    key: other_text,
                },
            )
            await check_replies(other_text, custom=True)

            stand = nsu
            await refresh()
            response = await http.get(path)
            assert response.status_code == 200
            setting = next(s for s in response.json()["settings"] if s["key"] == key)
            assert setting["kind"] == "textarea"
            assert setting["value"] == nsu_text
            await check_replies(nsu_text, custom=True)
            await save(1, {key: ""})
            await check_replies("«НГУ»", custom=False)

            stand = other
            await refresh()
            await check_replies(other_text, custom=True)
            await save(1, {key: ""})
            await check_replies("«СибГИУ»", custom=False)
    finally:
        async with other_engine.begin() as connection:
            await connection.execute(text("DROP SCHEMA welcome_other_stand CASCADE"))
