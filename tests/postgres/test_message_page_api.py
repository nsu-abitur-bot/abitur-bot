from datetime import datetime

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.auth.dependencies import get_current_admin
from api.main import app
from db.postgres.db import get_async_session
from db.postgres.models import Admin, Message, User


@pytest_asyncio.fixture
async def message_client(session):
    async def database():
        yield session

    app.dependency_overrides[get_async_session] = database
    app.dependency_overrides[get_current_admin] = lambda: Admin(username="reader")
    async with AsyncClient(
        transport=ASGITransport(app), base_url="http://test"
    ) as client:
        yield client
    app.dependency_overrides.pop(get_async_session, None)
    app.dependency_overrides.pop(get_current_admin, None)


async def seed(session):
    session.add_all([User(user_id=11, telegram_id=101), User(user_id=22, max_id="202")])
    await session.flush()
    session.add_all(
        [
            Message(
                id="a",
                user_id=11,
                session_id="session-first",
                user_text="[from Alice] Приём 100%_готов",
                bot_response="НГУ",
                created_at=datetime(2026, 11, 1),
            ),
            Message(
                id="b",
                user_id=11,
                session_id="session-second",
                user_text="[from Bob] Вопрос",
                bot_response="Бюджет",
                created_at=datetime(2026, 11, 1, 23, 59, 59, 999999),
            ),
            Message(
                id="c",
                user_id=22,
                session_id="session-third",
                user_text="Другой",
                bot_response="Контракт",
                created_at=datetime(2026, 11, 2),
            ),
            Message(
                id="d",
                user_id=22,
                session_id="session-fourth",
                user_text="Последний",
                bot_response="НГУ",
                created_at=datetime(2026, 11, 2),
            ),
        ]
    )
    await session.commit()


@pytest.mark.asyncio
async def test_page_total_and_stable_order(message_client, session):
    await seed(session)
    first = await message_client.get("/api/v1/messages/page", params={"limit": 1})
    assert first.status_code == 200
    assert first.json()["total"] == 4
    assert first.json()["limit"] == 1
    assert first.json()["offset"] == 0
    assert [m["id"] for m in first.json()["items"]] == ["d"]
    second = await message_client.get(
        "/api/v1/messages/page", params={"limit": 1, "offset": 1}
    )
    assert [m["id"] for m in second.json()["items"]] == ["c"]
    beyond = await message_client.get("/api/v1/messages/page", params={"offset": 500})
    assert beyond.json()["items"] == []
    assert beyond.json()["total"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "search, ids",
    [
        ("ALICE", ["a"]),
        ("бюдЖЕТ", ["b"]),
        ("session-third", ["c"]),
        ("11", ["b", "a"]),
        ("%_", ["a"]),
        ("no match", []),
    ],
)
async def test_search_all_fields_literal(message_client, session, search, ids):
    await seed(session)
    response = await message_client.get(
        "/api/v1/messages/page", params={"search": search}
    )
    assert response.status_code == 200
    assert [m["id"] for m in response.json()["items"]] == ids
    assert response.json()["total"] == len(ids)


@pytest.mark.asyncio
async def test_combined_filters_and_inclusive_dates(message_client, session):
    await seed(session)
    response = await message_client.get(
        "/api/v1/messages/page",
        params={
            "from_date": "2026-11-01",
            "to_date": "2026-11-01",
            "channel": "telegram",
            "user_id": 11,
        },
    )
    assert response.status_code == 200
    assert response.json()["total"] == 2
    assert [m["id"] for m in response.json()["items"]] == ["b", "a"]
    assert response.json()["items"][1]["username"] == "Alice"
    assert response.json()["items"][1]["user_text"] == "Приём 100%_готов"
    assert response.json()["items"][1]["messenger"] == "Telegram"
    max_page = await message_client.get(
        "/api/v1/messages/page", params={"channel": "MAX", "search": "НГУ"}
    )
    assert max_page.json()["total"] == 1
    assert max_page.json()["items"][0]["id"] == "d"


@pytest.mark.asyncio
async def test_page_beyond_first_500(message_client, session):
    session.add(User(user_id=11, telegram_id=101))
    await session.flush()
    session.add_all(
        [
            Message(
                id=f"{i:04}",
                user_id=11,
                session_id="s",
                user_text="oldest needle" if i == 0 else "question",
                bot_response="answer",
                created_at=datetime(2026, 11, 1),
            )
            for i in range(501)
        ]
    )
    await session.commit()
    response = await message_client.get("/api/v1/messages/page", params={"offset": 500})
    assert response.json()["total"] == 501
    assert response.json()["items"][0]["id"] == "0000"
    searched = await message_client.get(
        "/api/v1/messages/page", params={"search": "oldest needle"}
    )
    assert searched.json()["total"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=501",
        "offset=-1",
        "from_date=oops",
        "from_date=2026-11-02&to_date=2026-11-01",
        "channel=unknown",
    ],
)
async def test_invalid_page_query(message_client, query):
    response = await message_client.get("/api/v1/messages/page?" + query)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_page_requires_auth(message_client):
    app.dependency_overrides.pop(get_current_admin)
    response = await message_client.get("/api/v1/messages/page")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_empty_page_and_legacy_response(message_client, session):
    empty = await message_client.get("/api/v1/messages/page")
    assert empty.json() == {
        "items": [],
        "total": 0,
        "limit": 25,
        "offset": 0,
        "snapshot_id": None,
    }
    await seed(session)
    legacy = await message_client.get("/api/v1/messages", params={"user_id": 11})
    assert legacy.status_code == 200
    assert isinstance(legacy.json(), list)
    assert len(legacy.json()) == 2


@pytest.mark.asyncio
async def test_page_database_failure_is_not_empty_history(message_client, session):
    from sqlalchemy import text

    await session.execute(text("DROP TABLE message"))
    await session.commit()
    response = await message_client.get("/api/v1/messages/page")
    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}
    await session.rollback()


@pytest.mark.asyncio
async def test_export_snapshot_excludes_new_messages_between_pages(
    message_client, session
):
    session.add(User(user_id=11, telegram_id=101))
    await session.flush()
    original = [
        Message(
            user_id=11,
            session_id="old",
            user_text="Original question",
            bot_response="Answer",
            created_at=datetime(2026, 11, 2),
        )
        for _ in range(4)
    ]
    session.add_all(original)
    await session.commit()
    first = (
        await message_client.get("/api/v1/messages/page", params={"limit": 2})
    ).json()
    snapshot = first["snapshot_id"]
    inserted = Message(
        user_id=11,
        session_id="new",
        user_text="Inserted backdated",
        bot_response="answer",
        created_at=datetime(2026, 11, 1, 12),
    )
    session.add(inserted)
    await session.commit()
    assert inserted.id > snapshot
    second = (
        await message_client.get(
            "/api/v1/messages/page",
            params={"limit": 2, "offset": 2, "snapshot_id": snapshot},
        )
    ).json()
    assert first["total"] == second["total"] == 4
    assert [m["id"] for m in first["items"] + second["items"]] == sorted(
        [m.id for m in original], reverse=True
    )
    current = (await message_client.get("/api/v1/messages/page")).json()
    assert current["total"] == 5
