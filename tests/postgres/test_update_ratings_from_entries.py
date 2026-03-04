"""
Тесты для RatingService.update_ratings_from_entries.

Новая логика (многопользовательская поддержка):
- Один applicant_id может быть связан с несколькими пользователями
- Для каждого пользователя с данным applicant_id создается/обновляется рейтинг
- Схема позволяет несколько пользователей с одним и тем же applicant_id
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.dto import RatingEntry
from db.postgres.services.rating import RatingService
from db.postgres.services.user import UserService

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


async def _make_user(user_service: UserService, user_id: int, applicant_id: str):
    """Создаёт пользователя и возвращает его."""
    return await user_service.create_user(user_id=user_id, applicant_id=applicant_id)


# ---------------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_creates_rating_for_single_user(session: AsyncSession):
    """Если один пользователь найден по applicant_id — создаётся новая запись
    UserRating."""
    user_service = UserService(session)
    rating_service = RatingService(session)

    await _make_user(user_service, user_id=1001, applicant_id="AAA0001")
    leaderboard = await rating_service.get_or_create_leaderboard(
        "https://example.com/lb1"
    )

    entries = [
        RatingEntry(
            identifier="AAA0001",
            place=7,
            status="Подано",
            competition_type="Общий конкурс",
        )
    ]

    stats = await rating_service.update_ratings_from_entries(leaderboard.id, entries)

    assert stats["created"] == 1
    assert stats["updated"] == 0
    assert stats["skipped"] == 0

    ratings = await rating_service.get_user_ratings(user_id=1001)
    assert len(ratings) == 1
    assert ratings[0].place == 7
    assert ratings[0].competition_type == "Общий конкурс"
    assert ratings[0].status == "Подано"


@pytest.mark.asyncio
async def test_creates_ratings_for_multiple_users_with_same_applicant_id(
    session: AsyncSession,
):
    """Если несколько пользователей имеют один applicant_id.

    Создаются записи для всех.
    """
    user_service = UserService(session)
    rating_service = RatingService(session)

    # Два пользователя подписаны на один и тот же applicant_id
    await _make_user(user_service, user_id=1101, applicant_id="SHA0001")
    await _make_user(user_service, user_id=1102, applicant_id="SHA0001")
    leaderboard = await rating_service.get_or_create_leaderboard(
        "https://example.com/lb_multi1"
    )

    entries = [
        RatingEntry(
            identifier="SHA0001",
            place=5,
            status="Подано",
            competition_type="Общий конкурс",
        )
    ]

    stats = await rating_service.update_ratings_from_entries(leaderboard.id, entries)

    # Должны быть созданы рейтинги для ОБОИХ пользователей
    assert stats["created"] == 2
    assert stats["updated"] == 0
    assert stats["skipped"] == 0

    ratings1 = await rating_service.get_user_ratings(user_id=1101)
    assert len(ratings1) == 1
    assert ratings1[0].place == 5

    ratings2 = await rating_service.get_user_ratings(user_id=1102)
    assert len(ratings2) == 1
    assert ratings2[0].place == 5


@pytest.mark.asyncio
async def test_updates_existing_rating(session: AsyncSession):
    """Если запись UserRating уже есть — обновляется место, competition_type и
    status."""
    user_service = UserService(session)
    rating_service = RatingService(session)

    await _make_user(user_service, user_id=1002, applicant_id="BBB0002")
    leaderboard = await rating_service.get_or_create_leaderboard(
        "https://example.com/lb2"
    )

    # Создаём начальную запись
    await rating_service.create_or_update_user_rating(
        user_id=1002,
        leaderboard_id=leaderboard.id,
        place=15,
        competition_type="Квота",
        status="Подано",
    )

    entries = [
        RatingEntry(
            identifier="BBB0002",
            place=8,
            status="Оригинал",
            competition_type="Общий конкурс",
        )
    ]

    stats = await rating_service.update_ratings_from_entries(leaderboard.id, entries)

    assert stats["created"] == 0
    assert stats["updated"] == 1
    assert stats["skipped"] == 0

    ratings = await rating_service.get_user_ratings(user_id=1002)
    assert len(ratings) == 1
    assert ratings[0].place == 8
    assert ratings[0].competition_type == "Общий конкурс"
    assert ratings[0].status == "Оригинал"


@pytest.mark.asyncio
async def test_updates_all_users_with_same_applicant_id(session: AsyncSession):
    """Если несколько пользователей имеют один applicant_id.

    Обновляется рейтинг для всех.
    """
    user_service = UserService(session)
    rating_service = RatingService(session)

    # Два пользователя подписаны на один и тот же applicant_id
    await _make_user(user_service, user_id=1201, applicant_id="SHA0002")
    await _make_user(user_service, user_id=1202, applicant_id="SHA0002")
    leaderboard = await rating_service.get_or_create_leaderboard(
        "https://example.com/lb_multi2"
    )

    # Создаём начальные записи для обоих
    await rating_service.create_or_update_user_rating(
        user_id=1201, leaderboard_id=leaderboard.id, place=100
    )
    await rating_service.create_or_update_user_rating(
        user_id=1202, leaderboard_id=leaderboard.id, place=200
    )

    # Обновляем через update_ratings_from_entries
    entries = [
        RatingEntry(
            identifier="SHA0002",
            place=25,
            status="Зачислен",
            competition_type="Целевой",
        )
    ]

    stats = await rating_service.update_ratings_from_entries(leaderboard.id, entries)

    # Должны быть обновлены оба рейтинга
    assert stats["created"] == 0
    assert stats["updated"] == 2
    assert stats["skipped"] == 0

    ratings1 = await rating_service.get_user_ratings(user_id=1201)
    assert ratings1[0].place == 25
    assert ratings1[0].status == "Зачислен"

    ratings2 = await rating_service.get_user_ratings(user_id=1202)
    assert ratings2[0].place == 25
    assert ratings2[0].competition_type == "Целевой"


@pytest.mark.asyncio
async def test_handles_mixed_case_multiple_users(session: AsyncSession):
    """Тест с несколькими applicant_id, где некоторые имеют несколько пользователей."""
    user_service = UserService(session)
    rating_service = RatingService(session)

    # applicant_id "MULTI" имеет 2 пользователей
    await _make_user(user_service, user_id=1301, applicant_id="MULTI")
    await _make_user(user_service, user_id=1302, applicant_id="MULTI")
    # applicant_id "SINGLE" имеет 1 пользователя
    await _make_user(user_service, user_id=1303, applicant_id="SINGLE")

    leaderboard = await rating_service.get_or_create_leaderboard(
        "https://example.com/lb_mixed"
    )

    entries = [
        RatingEntry(identifier="MULTI", place=10, status="", competition_type=""),
        RatingEntry(identifier="SINGLE", place=20, status="", competition_type=""),
        RatingEntry(identifier="NOTFOUND", place=30, status="", competition_type=""),
    ]

    stats = await rating_service.update_ratings_from_entries(leaderboard.id, entries)

    # Создаём 3 рейтинга: 2 для MULTI, 1 для SINGLE
    assert stats["created"] == 3
    assert stats["updated"] == 0
    assert stats["skipped"] == 1

    r1 = await rating_service.get_user_ratings(user_id=1301)
    assert r1[0].place == 10

    r2 = await rating_service.get_user_ratings(user_id=1302)
    assert r2[0].place == 10

    r3 = await rating_service.get_user_ratings(user_id=1303)
    assert r3[0].place == 20


@pytest.mark.asyncio
async def test_skips_unknown_identifier(session: AsyncSession):
    """Если пользователь не найден по applicant_id — запись пропускается."""
    rating_service = RatingService(session)
    leaderboard = await rating_service.get_or_create_leaderboard(
        "https://example.com/lb3"
    )

    entries = [
        RatingEntry(
            identifier="UNKNOWN",
            place=1,
            status="",
            competition_type="Общий конкурс",
        )
    ]

    stats = await rating_service.update_ratings_from_entries(leaderboard.id, entries)

    assert stats["created"] == 0
    assert stats["updated"] == 0
    assert stats["skipped"] == 1


@pytest.mark.asyncio
async def test_skips_empty_identifier(session: AsyncSession):
    """Если identifier пустой — запись пропускается."""
    rating_service = RatingService(session)
    leaderboard = await rating_service.get_or_create_leaderboard(
        "https://example.com/lb_empty_id"
    )

    entries = [
        RatingEntry(
            identifier="",
            place=1,
            status="",
            competition_type="",
        ),
        RatingEntry(
            identifier="",
            place=2,
            status="",
            competition_type="",
        ),
    ]

    stats = await rating_service.update_ratings_from_entries(leaderboard.id, entries)

    assert stats["created"] == 0
    assert stats["updated"] == 0
    assert stats["skipped"] == 2


@pytest.mark.asyncio
async def test_mixed_entries(session: AsyncSession):
    """Проверяет корректную обработку смешанного списка: известные и неизвестные."""
    user_service = UserService(session)
    rating_service = RatingService(session)

    await _make_user(user_service, user_id=2001, applicant_id="CCC2001")
    await _make_user(user_service, user_id=2002, applicant_id="CCC2002")
    leaderboard = await rating_service.get_or_create_leaderboard(
        "https://example.com/lb4"
    )

    # Для user 2002 уже есть запись
    await rating_service.create_or_update_user_rating(
        user_id=2002, leaderboard_id=leaderboard.id, place=50
    )

    entries = [
        RatingEntry(
            identifier="CCC2001", place=3, competition_type="Общий", status="Подано"
        ),
        RatingEntry(
            identifier="CCC2002",
            place=10,
            competition_type="Целевой",
            status="Оригинал",
        ),
        RatingEntry(identifier="UNKNOWN", place=1, competition_type="Общий", status=""),
    ]

    stats = await rating_service.update_ratings_from_entries(leaderboard.id, entries)

    assert stats["created"] == 1
    assert stats["updated"] == 1
    assert stats["skipped"] == 1

    r1 = await rating_service.get_user_ratings(user_id=2001)
    assert r1[0].place == 3
    assert r1[0].competition_type == "Общий"
    assert r1[0].status == "Подано"

    r2 = await rating_service.get_user_ratings(user_id=2002)
    assert r2[0].place == 10
    assert r2[0].competition_type == "Целевой"
    assert r2[0].status == "Оригинал"


@pytest.mark.asyncio
async def test_empty_entries_list(session: AsyncSession):
    """Пустой список не вызывает ошибок и возвращает нулевую статистику."""
    rating_service = RatingService(session)
    leaderboard = await rating_service.get_or_create_leaderboard(
        "https://example.com/lb5"
    )

    stats = await rating_service.update_ratings_from_entries(leaderboard.id, [])

    assert stats == {"created": 0, "updated": 0, "skipped": 0}


@pytest.mark.asyncio
async def test_competition_type_and_status_stored_correctly(session: AsyncSession):
    """competition_type и status сохраняются точно как в RatingEntry."""
    user_service = UserService(session)
    rating_service = RatingService(session)

    await _make_user(user_service, user_id=3001, applicant_id="DDD3001")
    leaderboard = await rating_service.get_or_create_leaderboard(
        "https://example.com/lb6"
    )

    competition = "Места для поступающих по особой квоте"
    status = "Согласие на зачисление"
    entries = [
        RatingEntry(
            identifier="DDD3001",
            place=2,
            competition_type=competition,
            status=status,
        )
    ]

    await rating_service.update_ratings_from_entries(leaderboard.id, entries)

    ratings = await rating_service.get_user_ratings(user_id=3001)
    assert ratings[0].competition_type == competition
    assert ratings[0].status == status


@pytest.mark.asyncio
async def test_old_data_not_deleted(session: AsyncSession):
    """
    Записи других лидербордов для того же пользователя остаются нетронутыми
    после обновления.
    """
    user_service = UserService(session)
    rating_service = RatingService(session)

    await _make_user(user_service, user_id=4001, applicant_id="EEE4001")
    lb1 = await rating_service.get_or_create_leaderboard("https://example.com/hist1")
    lb2 = await rating_service.get_or_create_leaderboard("https://example.com/hist2")

    # Уже есть запись для lb1
    await rating_service.create_or_update_user_rating(
        user_id=4001,
        leaderboard_id=lb1.id,
        place=20,
        competition_type="Старый тип",
        status="Подано",
    )

    # Парсер обновляет lb2
    entries = [
        RatingEntry(
            identifier="EEE4001",
            place=5,
            competition_type="Новый тип",
            status="Оригинал",
        )
    ]
    await rating_service.update_ratings_from_entries(lb2.id, entries)

    all_ratings = await rating_service.get_user_ratings(user_id=4001)
    assert len(all_ratings) == 2

    lb1_rating = next(r for r in all_ratings if r.leaderboard_id == lb1.id)
    assert lb1_rating.place == 20
    assert lb1_rating.competition_type == "Старый тип"
    assert lb1_rating.status == "Подано"

    lb2_rating = next(r for r in all_ratings if r.leaderboard_id == lb2.id)
    assert lb2_rating.place == 5
    assert lb2_rating.competition_type == "Новый тип"
    assert lb2_rating.status == "Оригинал"
