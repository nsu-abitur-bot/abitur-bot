import logging
from datetime import UTC, datetime
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from parser.rating_parser import RatingEntry

from ..models import Leaderboard, User, UserRating

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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
                leaderboard.updated_at = datetime.now(UTC).replace(tzinfo=None)
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
        self,
        user_id: int,
        leaderboard_id: str,
        place: int,
        competition_type: str = "",
        status: str = "",
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
            user_rating = UserRating(
                user_id=user_id,
                leaderboard_id=leaderboard_id,
                place=place,
                competition_type=competition_type,
                status=status,
            )
            self.session.add(user_rating)
            logger.info(
                f"Создана новая запись рейтинга для пользователя {user_id}, место: {place}"  # noqa: E501
            )
        else:
            old_place = user_rating.place
            user_rating.place = place
            user_rating.competition_type = competition_type
            user_rating.status = status
            logger.info(
                f"Обновлена позиция пользователя {user_id}: {old_place} -> {place}"
            )

        await self.session.commit()
        await self.session.refresh(user_rating)
        return user_rating

    async def update_ratings_from_entries(
        self,
        leaderboard_id: str,
        entries: List[RatingEntry],
    ) -> dict:
        stats = {"updated": 0, "created": 0, "skipped": 0}

        for entry in entries:
            # Ищем пользователя по идентификатору абитуриента
            # (identifier из RatingEntry)
            user_result = await self.session.execute(
                select(User).where(User.applicant_id == entry.identifier)
            )
            user = user_result.scalar_one_or_none()

            if user is None:
                logger.debug(
                    f"Абитуриент с идентификатором '{entry.identifier}' "
                    f"не найден в БД — пропуск"
                )
                stats["skipped"] += 1
                continue

            # Ищем существующую запись рейтинга
            rating_result = await self.session.execute(
                select(UserRating).where(
                    UserRating.user_id == user.user_id,
                    UserRating.leaderboard_id == leaderboard_id,
                )
            )
            user_rating = rating_result.scalar_one_or_none()

            if user_rating is None:
                user_rating = UserRating(
                    user_id=user.user_id,
                    leaderboard_id=leaderboard_id,
                    place=entry.place,
                    competition_type=entry.competition_type,
                    status=entry.status,
                )
                self.session.add(user_rating)
                stats["created"] += 1
                logger.info(
                    f"Создана запись рейтинга: user={user.user_id}, "
                    f"place={entry.place}, "
                    f"competition_type='{entry.competition_type}', "
                    f"status='{entry.status}'"
                )
            else:
                old_place = user_rating.place
                user_rating.place = entry.place
                user_rating.competition_type = entry.competition_type
                user_rating.status = entry.status
                user_rating.updated_at = datetime.now(UTC).replace(tzinfo=None)
                stats["updated"] += 1
                logger.info(
                    f"Обновлена запись рейтинга: user={user.user_id}, "
                    f"{old_place} -> {entry.place}, "
                    f"competition_type='{entry.competition_type}', "
                    f"status='{entry.status}'"
                )

        try:
            await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка при сохранении обновлений рейтинга: {e}")
            raise

        logger.info(
            f"update_ratings_from_entries: создано={stats['created']}, "
            f"обновлено={stats['updated']}, пропущено={stats['skipped']}"
        )
        return stats

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
