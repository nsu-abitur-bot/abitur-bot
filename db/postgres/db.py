from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import DATABASE_CONFIG, DATABASE_URL

# Создаем движок
engine = create_async_engine(DATABASE_URL, **DATABASE_CONFIG)

# Создаем фабрику сессий
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Получение сессии БД."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Alias for FastAPI dependency
get_db = get_async_session
