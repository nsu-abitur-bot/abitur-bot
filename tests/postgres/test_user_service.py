"""
Тесты для UserService.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.services.user import UserService


@pytest.mark.asyncio
async def test_create_user(session: AsyncSession):
    """Тест создания пользователя."""
    service = UserService(session)

    user = await service.create_user(user_id=123456, snils="123-456-789 00")

    assert user is not None
    assert user.user_id == 123456
    assert user.snils_id == "123-456-789 00"


@pytest.mark.asyncio
async def test_create_duplicate_user(session: AsyncSession):
    """Тест создания дубликата пользователя."""
    service = UserService(session)

    # Создаём первого пользователя
    user1 = await service.create_user(user_id=123456, snils="123-456-789 00")
    assert user1 is not None

    # Очищаем сессию перед повторной попыткой
    await session.rollback()
    session.expunge_all()

    # Пытаемся создать дубликат
    user2 = await service.create_user(user_id=123456, snils="999-999-999 99")
    assert user2 is None  # Должен вернуть None


@pytest.mark.asyncio
async def test_get_user(session: AsyncSession):
    """Тест получения пользователя."""
    service = UserService(session)

    # Создаём пользователя
    created_user = await service.create_user(user_id=123456, snils="123-456-789 00")
    assert created_user is not None

    # Получаем пользователя
    user = await service.get_user(user_id=123456)
    assert user is not None
    assert user.user_id == 123456
    assert user.snils_id == "123-456-789 00"


@pytest.mark.asyncio
async def test_get_nonexistent_user(session: AsyncSession):
    """Тест получения несуществующего пользователя."""
    service = UserService(session)

    user = await service.get_user(user_id=999999)
    assert user is None


@pytest.mark.asyncio
async def test_get_user_by_snils(session: AsyncSession):
    """Тест получения пользователя по SNILS."""
    service = UserService(session)

    # Создаём пользователя
    await service.create_user(user_id=123456, snils="123-456-789 00")

    # Получаем по SNILS
    user = await service.get_user_by_snils(snils="123-456-789 00")
    assert user is not None
    assert user.user_id == 123456


@pytest.mark.asyncio
async def test_get_all_users(session: AsyncSession):
    """Тест получения всех пользователей."""
    service = UserService(session)

    # Создаём несколько пользователей
    await service.create_user(user_id=111111, snils="111-111-111 11")
    await service.create_user(user_id=222222, snils="222-222-222 22")
    await service.create_user(user_id=333333, snils="333-333-333 33")

    users = await service.get_all_users()
    assert len(users) == 3


@pytest.mark.asyncio
async def test_user_exists(session: AsyncSession):
    """Тест проверки существования пользователя."""
    service = UserService(session)

    # Создаём пользователя
    await service.create_user(user_id=123456, snils="123-456-789 00")

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
    await service.create_user(user_id=111111, snils="111-111-111 11")
    await service.create_user(user_id=222222, snils="222-222-222 22")

    count = await service.get_user_count()
    assert count == 2


@pytest.mark.asyncio
async def test_update_snils(session: AsyncSession):
    """Тест обновления SNILS."""
    service = UserService(session)

    # Создаём пользователя
    await service.create_user(user_id=123456, snils="123-456-789 00")

    # Обновляем SNILS
    success = await service.update_snils(user_id=123456, new_snils="999-999-999 99")
    assert success is True

    # Проверяем обновление
    user = await service.get_user(user_id=123456)
    assert user.snils_id == "999-999-999 99"  # type: ignore


@pytest.mark.asyncio
async def test_delete_user(session: AsyncSession):
    """Тест удаления пользователя."""
    service = UserService(session)

    # Создаём пользователя
    await service.create_user(user_id=123456, snils="123-456-789 00")

    # Удаляем
    success = await service.delete_user(user_id=123456)
    assert success is True

    # Проверяем что пользователь удалён
    user = await service.get_user(user_id=123456)
    assert user is None
