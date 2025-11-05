import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import DATABASE_CONFIG, DATABASE_URL
from .models import Base

logger = logging.getLogger(__name__)

# Создаем движок
engine = create_async_engine(DATABASE_URL, **DATABASE_CONFIG)

# Создаем фабрику сессий
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    """Инициализация базы данных (создание таблиц)."""
    logger.info("Начало инициализации базы данных...")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("✅ База данных инициализирована")


async def drop_db():
    """Удаление всех таблиц (для тестов)."""
    logger.warning("Удаление всех таблиц...")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    logger.info("✅ Все таблицы удалены")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Получение сессии БД."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
