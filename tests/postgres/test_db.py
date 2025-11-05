import asyncio
import logging
import os
import sys
from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Добавляем корневую папку проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from db.postgres import (
    DATABASE_CONFIG,
    DATABASE_URL,
    Base,
    RatingService,
    UserService,
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestDatabaseBasics:
    """Базовые тесты БД без сложных фикстур."""

    @pytest.mark.asyncio
    async def test_database_connection(self):
        """Тест подключения к БД."""
        # Создаем временный engine для теста
        engine = create_async_engine(DATABASE_URL, **DATABASE_CONFIG)

        try:
            async with engine.begin() as conn:
                result = await conn.execute(text("SELECT 1 as test_value"))
                value = result.scalar()
                assert value == 1
                logger.info("✅ Подключение к БД успешно")
        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_table_creation(self):
        """Тест создания таблиц."""
        engine = create_async_engine(DATABASE_URL, **DATABASE_CONFIG)

        try:
            # Удаляем и создаем таблицы
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)

            # Проверяем, что таблицы существуют
            async with engine.begin() as conn:
                tables_query = text("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                result = await conn.execute(tables_query)
                tables = [row[0] for row in result.fetchall()]

                expected_tables = {"users", "leaderboards", "user_ratings"}
                actual_tables = set(tables)

                assert expected_tables.issubset(actual_tables), (
                    f"Отсутствуют таблицы: {expected_tables - actual_tables}"
                )
                logger.info(f"✅ Таблицы созданы: {sorted(tables)}")
        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_user_service_basic(self):
        """Базовый тест UserService."""
        engine = create_async_engine(DATABASE_URL, **DATABASE_CONFIG)
        session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        try:
            # Настройка БД
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)

            # Создание пользователя
            async with session_factory() as session:
                user_service = UserService(session)
                user = await user_service.create_user(12345, "123-456-789 00")
                assert user is not None
                assert user.user_id == 12345
                assert user.snils_id == "123-456-789 00"
                logger.info("✅ Пользователь создан успешно")

                # ИСПРАВЛЕНИЕ: Сохраняем только ID для проверки
                user_id = user.user_id
                # Отсоединяем объект от сессии
                session.expunge(user)

            # Получение пользователя в новой сессии
            async with session_factory() as session:
                user_service = UserService(session)
                retrieved_user = await user_service.get_user(user_id)
                assert retrieved_user is not None
                assert retrieved_user.user_id == user_id
                logger.info("✅ Пользователь получен успешно")

            # Тест несуществующего пользователя
            async with session_factory() as session:
                user_service = UserService(session)
                no_user = await user_service.get_user(99999)
                assert no_user is None
                logger.info("✅ Несуществующий пользователь обработан корректно")

            # Тест дублирующегося пользователя
            async with session_factory() as session:
                user_service = UserService(session)
                duplicate_user = await user_service.create_user(
                    user_id, "999-999-999 99"
                )
                assert duplicate_user is None
                logger.info("✅ Дублирующийся пользователь обработан корректно")

            # Тест подсчета пользователей
            async with session_factory() as session:
                user_service = UserService(session)
                count = await user_service.get_user_count()
                assert count == 1
                logger.info(f"✅ Количество пользователей: {count}")

        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_rating_service_basic(self):
        """Базовый тест RatingService."""
        engine = create_async_engine(DATABASE_URL, **DATABASE_CONFIG)
        session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        try:
            # Настройка БД
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)

            # Тест операций с рейтингами
            async with session_factory() as session:
                rating_service = RatingService(session)

                # Создание лидерборда
                url = "https://example.com/leaderboard"
                leaderboard = await rating_service.get_or_create_leaderboard(url)
                assert leaderboard is not None
                assert leaderboard.url == url
                logger.info("✅ Лидерборд создан успешно")

                # Сохраняем ID для следующей проверки
                leaderboard_id = leaderboard.id
                session.expunge(leaderboard)

            # Получение того же лидерборда в новой сессии
            async with session_factory() as session:
                rating_service = RatingService(session)
                same_leaderboard = await rating_service.get_or_create_leaderboard(url)
                assert same_leaderboard.id == leaderboard_id
                logger.info("✅ Получение лидерборда работает")

        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_user_rating_integration(self):
        """Интеграционный тест пользователей и рейтингов."""
        engine = create_async_engine(DATABASE_URL, **DATABASE_CONFIG)
        session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        try:
            # Настройка БД
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)

            # Создание пользователя и лидерборда
            async with session_factory() as session:
                user_service = UserService(session)
                rating_service = RatingService(session)

                # Создание пользователя
                user = await user_service.create_user(54321, "543-210-987 00")
                assert user is not None
                user_id = user.user_id
                logger.info("✅ Пользователь создан для интеграционного теста")

                # Создание лидерборда
                leaderboard = await rating_service.get_or_create_leaderboard(
                    "https://integration-test.com"
                )
                assert leaderboard is not None
                leaderboard_id = leaderboard.id
                logger.info("✅ Лидерборд создан для интеграционного теста")

                # Создание рейтинга пользователя
                rating = await rating_service.create_or_update_user_rating(
                    user_id, leaderboard_id, 3
                )
                assert rating is not None
                assert rating.user_id == user_id
                assert rating.place == 3
                logger.info("✅ Рейтинг пользователя создан успешно")

                # Обновление рейтинга пользователя
                updated_rating = await rating_service.create_or_update_user_rating(
                    user_id, leaderboard_id, 1
                )
                assert updated_rating.place == 1
                logger.info("✅ Рейтинг пользователя обновлен успешно")

                # Получение рейтингов пользователя
                user_ratings = await rating_service.get_user_ratings(user_id)
                assert len(user_ratings) == 1
                assert user_ratings[0].place == 1
                logger.info("✅ Рейтинги пользователя получены успешно")

        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_multiple_users_and_ratings(self):
        """Тест множественных пользователей и рейтингов."""
        engine = create_async_engine(DATABASE_URL, **DATABASE_CONFIG)
        session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        try:
            # Настройка БД
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)

            async with session_factory() as session:
                user_service = UserService(session)
                rating_service = RatingService(session)

                # Создание нескольких пользователей с правильным форматом СНИЛС
                users_data = [
                    (1001, "100-100-100 01"),
                    (1002, "200-200-200 02"),
                    (1003, "300-300-300 03"),
                ]

                created_user_ids = []
                for user_id, snils in users_data:
                    user = await user_service.create_user(user_id, snils)
                    assert user is not None
                    created_user_ids.append(user.user_id)

                logger.info(f"✅ Создано {len(created_user_ids)} пользователей")

                # Создание нескольких лидербордов
                urls = ["https://test1.com", "https://test2.com", "https://test3.com"]

                created_leaderboard_ids = []
                for url in urls:
                    leaderboard = await rating_service.get_or_create_leaderboard(url)
                    assert leaderboard is not None
                    created_leaderboard_ids.append(leaderboard.id)

                logger.info(f"✅ Создано {len(created_leaderboard_ids)} лидербордов")

                # Создание рейтингов для каждого пользователя в каждом лидерборде
                rating_count = 0
                for i, user_id in enumerate(created_user_ids):
                    for j, leaderboard_id in enumerate(created_leaderboard_ids):
                        place = (i * len(created_leaderboard_ids)) + j + 1
                        rating = await rating_service.create_or_update_user_rating(
                            user_id, leaderboard_id, place
                        )
                        assert rating is not None
                        rating_count += 1

                logger.info(f"✅ Создано {rating_count} рейтингов")

                # Проверка существования всех пользователей
                all_users = await user_service.get_all_users()
                assert len(all_users) == 3
                logger.info("✅ Все пользователи получены успешно")

                # Проверка существования всех лидербордов
                all_leaderboards = await rating_service.get_all_leaderboards()
                assert len(all_leaderboards) == 3
                logger.info("✅ Все лидерборды получены успешно")

                # Проверка рейтингов пользователей
                for user_id in created_user_ids:
                    user_ratings = await rating_service.get_user_ratings(user_id)
                    assert (
                        len(user_ratings) == 3
                    )  # Каждый пользователь имеет рейтинги в 3 лидербордах

                logger.info("✅ Все рейтинги пользователей проверены успешно")

        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_performance_basic(self):
        """Базовый тест производительности."""
        engine = create_async_engine(DATABASE_URL, **DATABASE_CONFIG)
        session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        try:
            # Настройка БД
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)

            start_time = datetime.now(UTC)

            async with session_factory() as session:
                user_service = UserService(session)

                # Создание 10 пользователей с правильным форматом СНИЛС (XXX-XXX-XXX XX)
                for i in range(10):
                    # Правильный формат СНИЛС: ровно 14 символов
                    snils = f"{200 + i:03d}-{100 + i:03d}-{300 + i:03d} {i % 100:02d}"
                    user = await user_service.create_user(2000 + i, snils)
                    assert user is not None, (
                        f"Не удалось создать пользователя {2000 + i} с СНИЛС {snils}"
                    )

            end_time = datetime.now(UTC)
            duration = (end_time - start_time).total_seconds()

            # Проверка количества в новой сессии
            async with session_factory() as session:
                user_service = UserService(session)
                count = await user_service.get_user_count()
                assert count == 10

            logger.info(f"✅ Создано 10 пользователей за {duration:.2f} секунд")
            assert duration < 10.0  # Должно выполниться за 10 секунд

        finally:
            await engine.dispose()


# Простой запуск для отладки
if __name__ == "__main__":

    async def run_single_test():
        """Запуск одного теста для отладки."""
        test_instance = TestDatabaseBasics()

        try:
            logger.info("🧪 Запуск одиночного теста...")
            await test_instance.test_user_service_basic()
            logger.info("✅ Одиночный тест прошел!")
        except Exception as e:
            logger.error(f"❌ Одиночный тест провален: {e}")
            import traceback

            traceback.print_exc()
            raise

    asyncio.run(run_single_test())
