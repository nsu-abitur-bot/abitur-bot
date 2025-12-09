"""
Тесты для RatingService.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.services.rating import RatingService
from db.postgres.services.user import UserService


@pytest.mark.asyncio
async def test_get_or_create_leaderboard(session: AsyncSession):
    """Тест создания/получения рейтинга."""
    service = RatingService(session)

    # Создаём рейтинг
    leaderboard = await service.get_or_create_leaderboard(
        url="https://example.com/ratings"
    )
    assert leaderboard is not None
    assert leaderboard.url == "https://example.com/ratings"

    # Получаем существующий
    same_leaderboard = await service.get_or_create_leaderboard(
        url="https://example.com/ratings"
    )
    assert same_leaderboard.id == leaderboard.id


@pytest.mark.asyncio
async def test_update_leaderboard_hash(session: AsyncSession):
    """Тест обновления хэша рейтинга."""
    service = RatingService(session)

    # Создаём рейтинг
    leaderboard = await service.get_or_create_leaderboard(
        url="https://example.com/ratings"
    )

    # Обновляем хэш
    success = await service.update_leaderboard_hash(
        leaderboard_id=leaderboard.id, content_hash="abc123"
    )
    assert success is True

    # Проверяем обновление
    updated = await service.get_leaderboard_by_id(leaderboard.id)
    assert updated is not None
    assert updated.content_hash == "abc123"


@pytest.mark.asyncio
async def test_create_or_update_user_rating(session: AsyncSession):
    """Тест создания/обновления рейтинга пользователя."""
    user_service = UserService(session)
    rating_service = RatingService(session)

    # Создаём пользователя
    await user_service.create_user(user_id=123456, snils="123-456-789 00")

    # Создаём рейтинг
    leaderboard = await rating_service.get_or_create_leaderboard(
        url="https://example.com/ratings"
    )

    # Создаём запись рейтинга
    user_rating = await rating_service.create_or_update_user_rating(
        user_id=123456, leaderboard_id=leaderboard.id, place=10
    )
    assert user_rating.place == 10

    # Обновляем место
    updated_rating = await rating_service.create_or_update_user_rating(
        user_id=123456, leaderboard_id=leaderboard.id, place=5
    )
    assert updated_rating.place == 5


@pytest.mark.asyncio
async def test_get_user_ratings(session: AsyncSession):
    """Тест получения всех рейтингов пользователя."""
    user_service = UserService(session)
    rating_service = RatingService(session)

    # Создаём пользователя
    await user_service.create_user(user_id=123456, snils="123-456-789 00")

    # Создаём несколько рейтингов
    leaderboard1 = await rating_service.get_or_create_leaderboard(
        url="https://example.com/ratings1"
    )
    leaderboard2 = await rating_service.get_or_create_leaderboard(
        url="https://example.com/ratings2"
    )

    await rating_service.create_or_update_user_rating(
        user_id=123456, leaderboard_id=leaderboard1.id, place=10
    )
    await rating_service.create_or_update_user_rating(
        user_id=123456, leaderboard_id=leaderboard2.id, place=20
    )

    # Получаем все рейтинги
    ratings = await rating_service.get_user_ratings(user_id=123456)
    assert len(ratings) == 2


@pytest.mark.asyncio
async def test_delete_user_rating(session: AsyncSession):
    """Тест удаления записи рейтинга."""
    user_service = UserService(session)
    rating_service = RatingService(session)

    # Создаём пользователя и рейтинг
    await user_service.create_user(user_id=123456, snils="123-456-789 00")
    leaderboard = await rating_service.get_or_create_leaderboard(
        url="https://example.com/ratings"
    )

    await rating_service.create_or_update_user_rating(
        user_id=123456, leaderboard_id=leaderboard.id, place=10
    )

    # Удаляем
    success = await rating_service.delete_user_rating(
        user_id=123456, leaderboard_id=leaderboard.id
    )
    assert success is True

    # Проверяем удаление
    ratings = await rating_service.get_user_ratings(user_id=123456)
    assert len(ratings) == 0
