import logging
from datetime import UTC, datetime
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Leaderboard, UserRating

logger = logging.getLogger(__name__)


class RatingService:
    """Сервис для работы с рейтингами."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create_leaderboard(self, url: str) -> Leaderboard:
        """Получить существующий рейтинг или создать новый."""
        result = await self.session.execute(
            select(Leaderboard).where(Leaderboard.url == url)
        )
        leaderboard = result.scalar_one_or_none()

        if not leaderboard:
            leaderboard = Leaderboard(url=url)
            self.session.add(leaderboard)
            await self.session.commit()
            await self.session.refresh(leaderboard)
            logger.info(f"Создан новый рейтинг: {url}")

        return leaderboard

    async def update_leaderboard_hash(
        self, leaderboard_id: str, content_hash: str
    ) -> bool:
        """Обновить хэш рейтинга."""
        try:
            result = await self.session.execute(
                select(Leaderboard).where(Leaderboard.id == leaderboard_id)
            )
            leaderboard = result.scalar_one_or_none()

            if leaderboard:
                leaderboard.content_hash = content_hash
                leaderboard.updated_at = datetime.now(UTC)
                await self.session.commit()
                logger.info(f"Обновлен хэш рейтинга {leaderboard_id}")
                return True
            logger.warning(f"Рейтинг {leaderboard_id} не найден для обновления хэша")
            return False
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка обновления хэша рейтинга {leaderboard_id}: {e}")
            return False

    async def get_all_leaderboards(self) -> List[Leaderboard]:
        """Получить все рейтинги."""
        result = await self.session.execute(select(Leaderboard))
        leaderboards = list(result.scalars().all())
        logger.info(f"Получено {len(leaderboards)} рейтингов")
        return leaderboards

    async def create_or_update_user_rating(
        self, user_id: int, leaderboard_id: str, place: int
    ) -> UserRating:
        """Создать или обновить рейтинг пользователя."""
        result = await self.session.execute(
            select(UserRating).where(
                UserRating.user_id == user_id,
                UserRating.leaderboard_id == leaderboard_id,
            )
        )
        user_rating = result.scalar_one_or_none()

        if not user_rating:
            # Создаем новую запись
            user_rating = UserRating(
                user_id=user_id, leaderboard_id=leaderboard_id, place=place
            )
            self.session.add(user_rating)
            logger.info(
                f"Создана новая запись рейтинга для пользователя {user_id}, место: {place}"  # noqa: E501
            )
        else:
            # Обновляем существующую
            old_place = user_rating.place
            user_rating.place = place
            logger.info(
                f"Обновлена позиция пользователя {user_id}: {old_place} -> {place}"
            )

        await self.session.commit()
        await self.session.refresh(user_rating)
        return user_rating

    async def get_user_ratings(self, user_id: int) -> List[UserRating]:
        """Получить все рейтинги пользователя."""
        result = await self.session.execute(
            select(UserRating).where(UserRating.user_id == user_id)
        )
        return list(result.scalars().all())

    async def get_leaderboard_by_id(self, leaderboard_id: str) -> Optional[Leaderboard]:
        """Получить рейтинг по ID."""
        result = await self.session.execute(
            select(Leaderboard).where(Leaderboard.id == leaderboard_id)
        )
        return result.scalar_one_or_none()

    async def delete_user_rating(self, user_id: int, leaderboard_id: str) -> bool:
        """Удалить запись рейтинга пользователя."""
        try:
            result = await self.session.execute(
                select(UserRating).where(
                    UserRating.user_id == user_id,
                    UserRating.leaderboard_id == leaderboard_id,
                )
            )
            user_rating = result.scalar_one_or_none()

            if user_rating:
                await self.session.delete(user_rating)
                await self.session.commit()
                logger.info(f"Удалена запись рейтинга пользователя {user_id}")
                return True
            logger.warning("Запись рейтинга не найдена для удаления")
            return False
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка удаления записи рейтинга: {e}")
            return False
