import fakeredis.aioredis as fakeredis
import pytest
import pytest_asyncio

from db.redis_client import HISTORY_LIMIT, RedisClient


@pytest_asyncio.fixture
async def redis_client(monkeypatch):
    fake_redis = await fakeredis.FakeRedis(decode_responses=True)

    # подменяем redis.from_url внутри RedisClient
    monkeypatch.setattr("db.redis_client.redis.from_url", lambda *a, **kw: fake_redis)

    client = RedisClient()
    yield client
    await client.close()


@pytest.mark.asyncio
async def test_add_and_get_history(redis_client):
    session_id = "test-session"

    # Добавляем несколько сообщений
    for i in range(HISTORY_LIMIT + 5):  # > HISTORY_LIMIT чтобы проверить ltrim
        await redis_client.add_message(session_id, {"id": str(i), "text": f"msg-{i}"})

    history = await redis_client.get_history(session_id)
    assert len(history) == HISTORY_LIMIT
    assert history[-1]["text"] == f"msg-{HISTORY_LIMIT + 4}"


@pytest.mark.asyncio
async def test_set_and_get(redis_client):
    await redis_client.set("foo", "bar")
    value = await redis_client.get("foo")
    assert value == "bar"


@pytest.mark.asyncio
async def test_expire_sets_ttl(redis_client):
    key = "expiring"
    await redis_client.set(key, "value")
    await redis_client.client.expire(key, 10)
    ttl = await redis_client.client.ttl(key)
    assert ttl <= 10 and ttl > 0


@pytest.mark.asyncio
async def test_get_non_existing_key(redis_client):
    value = await redis_client.get("missing-key")
    assert value is None
