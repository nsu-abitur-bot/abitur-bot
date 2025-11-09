"""
Конфигурация pytest для тестирования с PostgreSQL.
"""

import os
from typing import AsyncGenerator

import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from db.postgres.models import Base

# Загружаем переменные окружения
load_dotenv()

# URL для тестовой БД
TEST_DB_NAME = os.getenv("TEST_DB_NAME", "abitur_test")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_user = os.getenv("DB_USER", "postgres")
db_password = os.getenv("DB_PASSWORD", "postgres")

TEST_DATABASE_URL = (
    f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{TEST_DB_NAME}"
)


@pytest_asyncio.fixture(scope="function")
async def test_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Создаём движок для тестовой БД."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        pool_size=1,
        max_overflow=0,
        pool_pre_ping=True,
    )

    # Создаём все таблицы
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Удаляем все таблицы
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture
async def session(test_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Создаём сессию для каждого теста."""
    async_session = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session
        await session.rollback()
