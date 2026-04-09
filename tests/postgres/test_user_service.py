"""
Тесты для UserService.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.models import User
from db.postgres.services.user import UserService


@pytest.mark.asyncio
async def test_create_user(session: AsyncSession):
    """Тест создания пользователя."""
    service = UserService(session)

    user = await service.create_user(user_id=123456, applicant_id="1234567")

    assert user is not None
    assert user.user_id == 123456
    assert user.applicant_id == "1234567"


@pytest.mark.asyncio
async def test_create_duplicate_user(session: AsyncSession):
    """Тест создания дубликата пользователя."""
    service = UserService(session)

    # Создаём первого пользователя
    user1 = await service.create_user(user_id=123456, applicant_id="1234567")
    assert user1 is not None

    # Очищаем сессию перед повторной попыткой
    await session.rollback()
    session.expunge_all()

    # Пытаемся создать дубликат
    user2 = await service.create_user(user_id=123456, applicant_id="9999999")
    assert user2 is None  # Должен вернуть None


@pytest.mark.asyncio
async def test_get_user(session: AsyncSession):
    """Тест получения пользователя."""
    service = UserService(session)

    # Создаём пользователя
    created_user = await service.create_user(user_id=123456, applicant_id="1234567")
    assert created_user is not None

    # Получаем пользователя
    user = await service.get_user(user_id=123456)
    assert user is not None
    assert user.user_id == 123456
    assert user.applicant_id == "1234567"


@pytest.mark.asyncio
async def test_get_nonexistent_user(session: AsyncSession):
    """Тест получения несуществующего пользователя."""
    service = UserService(session)

    user = await service.get_user(user_id=999999)
    assert user is None


@pytest.mark.asyncio
async def test_get_users_by_applicant_id(session: AsyncSession):
    """Тест получения пользователей по идентификатору абитуриента.

    Теперь несколько пользователей могут иметь один applicant_id.
    """
    service = UserService(session)

    # Создаём несколько пользователей с одним applicant_id
    await service.create_user(user_id=123456, applicant_id="1234567")
    await service.create_user(user_id=789012, applicant_id="1234567")

    # Получаем всех пользователей с этим applicant_id
    users = await service.get_users_by_applicant_id(applicant_id="1234567")
    assert len(users) == 2
    user_ids = {u.user_id for u in users}
    assert user_ids == {123456, 789012}


@pytest.mark.asyncio
async def test_get_all_users(session: AsyncSession):
    """Тест получения всех пользователей."""
    service = UserService(session)

    # Создаём несколько пользователей
    await service.create_user(user_id=111111, applicant_id="AAA1111")
    await service.create_user(user_id=222222, applicant_id="BBB2222")
    await service.create_user(user_id=333333, applicant_id="CCC3333")

    users = await service.get_all_users()
    assert len(users) == 3


@pytest.mark.asyncio
async def test_user_exists(session: AsyncSession):
    """Тест проверки существования пользователя."""
    service = UserService(session)

    # Создаём пользователя
    await service.create_user(user_id=123456, applicant_id="1234567")

    # Проверяем существование
    exists = await service.user_exists(user_id=123456)
    assert exists is True

    # Проверяем несуществующего
    not_exists = await service.user_exists(user_id=999999)
    assert not_exists is False


@pytest.mark.asyncio
async def test_get_user_count(session: AsyncSession):
    """Тест подсчета пользователей."""
    service = UserService(session)

    # Создаём несколько пользователей
    await service.create_user(user_id=111111, applicant_id="AAA1111")
    await service.create_user(user_id=222222, applicant_id="BBB2222")

    count = await service.get_user_count()
    assert count == 2


@pytest.mark.asyncio
async def test_update_applicant_id(session: AsyncSession):
    """Тест обновления идентификатора абитуриента."""
    service = UserService(session)

    # Создаём пользователя
    await service.create_user(user_id=123456, applicant_id="1234567")

    # Обновляем applicant_id
    success = await service.update_applicant_id(
        user_id=123456, new_applicant_id="7654321"
    )
    assert success is True

    # Проверяем обновление
    user = await service.get_user(user_id=123456)
    assert user.applicant_id == "7654321"  # type: ignore


@pytest.mark.asyncio
async def test_delete_user(session: AsyncSession):
    """Тест удаления пользователя."""
    service = UserService(session)

    # Создаём пользователя
    await service.create_user(user_id=123456, applicant_id="1234567")

    # Удаляем
    success = await service.delete_user(user_id=123456)
    assert success is True

    # Проверяем что пользователь удалён
    user = await service.get_user(user_id=123456)
    assert user is None


@pytest.mark.asyncio
async def test_get_user_count_stats_empty(session: AsyncSession):
    """Тест статистики при отсутствии пользователей."""
    service = UserService(session)

    stats = await service.get_user_count_stats()
    assert stats == {"day": 0, "week": 0, "month": 0, "year": 0, "all_time": 0}


@pytest.mark.asyncio
async def test_get_user_count_stats_all_recent(session: AsyncSession):
    """Тест: только что созданные пользователи попадают во все периоды."""
    service = UserService(session)

    await service.create_user(user_id=111111, applicant_id="AAA1111")
    await service.create_user(user_id=222222, applicant_id="BBB2222")

    stats = await service.get_user_count_stats()
    assert stats["day"] == 2
    assert stats["week"] == 2
    assert stats["month"] == 2
    assert stats["year"] == 2
    assert stats["all_time"] == 2


@pytest.mark.asyncio
async def test_get_user_count_stats_old_user_excluded_from_day(
    session: AsyncSession,
):
    """Тест: пользователь созданный 2 дня назад не попадает в статистику за день."""
    service = UserService(session)

    # Создаём пользователя
    user = await service.create_user(user_id=111111, applicant_id="AAA1111")
    assert user is not None

    # Сдвигаем created_at на 2 дня назад (но в пределах недели)
    now = datetime.now(UTC).replace(tzinfo=None)
    two_days_ago = now - timedelta(days=2)
    await session.execute(
        update(User).where(User.user_id == 111111).values(created_at=two_days_ago)
    )
    await session.commit()

    stats = await service.get_user_count_stats()
    assert stats["day"] == 0
    assert stats["week"] == 1
    assert stats["month"] == 1
    assert stats["year"] == 1
    assert stats["all_time"] == 1


@pytest.mark.asyncio
async def test_get_user_count_stats_boundary_365_days(session: AsyncSession):
    """Тест: пользователь созданный более 365 дней назад не попадает в 'year'."""
    service = UserService(session)

    user = await service.create_user(user_id=111111, applicant_id="AAA1111")
    assert user is not None

    now = datetime.now(UTC).replace(tzinfo=None)
    over_year_ago = now - timedelta(days=366)
    await session.execute(
        update(User).where(User.user_id == 111111).values(created_at=over_year_ago)
    )
    await session.commit()

    stats = await service.get_user_count_stats()
    assert stats["day"] == 0
    assert stats["week"] == 0
    assert stats["month"] == 0
    assert stats["year"] == 0
    assert stats["all_time"] == 1


@pytest.mark.asyncio
async def test_ensure_user_by_telegram_id_creates_new_user(session: AsyncSession):
    """Тест: ensure_user_by_telegram_id создает пользователя при отсутствии."""
    service = UserService(session)

    user = await service.ensure_user_by_telegram_id(telegram_id=5550001)

    assert user.telegram_id == 5550001
    assert user.max_id is None
    assert user.user_id is not None


@pytest.mark.asyncio
async def test_ensure_user_by_telegram_id_returns_existing(session: AsyncSession):
    """Тест: ensure_user_by_telegram_id возвращает существующего пользователя."""
    service = UserService(session)

    created = await service.create_user(user_id=200001, telegram_id=8881001)
    assert created is not None

    resolved = await service.ensure_user_by_telegram_id(telegram_id=8881001)
    assert resolved.user_id == 200001
    assert resolved.telegram_id == 8881001


@pytest.mark.asyncio
async def test_ensure_user_by_max_id_creates_new_user(session: AsyncSession):
    """Тест: ensure_user_by_max_id создает нового пользователя для MAX."""
    service = UserService(session)

    user = await service.ensure_user_by_max_id(max_id="max-user-42")

    assert user.max_id == "max-user-42"
    assert user.telegram_id is None
    assert user.user_id is not None


@pytest.mark.asyncio
async def test_get_user_by_channel_id(session: AsyncSession):
    """Тест резолва пользователя по channel/external_id."""
    service = UserService(session)

    created = await service.create_user(
        user_id=300001,
        telegram_id=7770001,
        max_id="max-777",
    )
    assert created is not None

    via_tg = await service.get_user_by_channel_id("telegram", "7770001")
    via_max = await service.get_user_by_channel_id("max", "max-777")

    assert via_tg is not None
    assert via_max is not None
    assert via_tg.user_id == 300001
    assert via_max.user_id == 300001


@pytest.mark.asyncio
async def test_bind_max_id(session: AsyncSession):
    """Тест привязки max_id к существующему пользователю."""
    service = UserService(session)

    created = await service.create_user(user_id=400001, telegram_id=9990001)
    assert created is not None

    ok = await service.bind_max_id(user_id=400001, max_id="max-400001")
    assert ok is True

    updated = await service.get_user(400001)
    assert updated is not None
    assert updated.max_id == "max-400001"
