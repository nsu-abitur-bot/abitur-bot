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
async def test_get_user_by_applicant_id(session: AsyncSession):
    """Тест получения пользователя по идентификатору абитуриента."""
    service = UserService(session)

    # Создаём пользователя
    await service.create_user(user_id=123456, applicant_id="1234567")

    # Получаем по applicant_id
    user = await service.get_user_by_applicant_id(applicant_id="1234567")
    assert user is not None
    assert user.user_id == 123456


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
