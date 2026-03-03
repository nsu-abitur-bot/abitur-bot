import logging
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..dto import RatingEntry
from ..models import Leaderboard, User, UserRating

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
                # updated_at обновляется автоматически через onupdate=timestamp
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

    # ------------------------------------------------------------------
    # Приватные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_entry_fields(rating: UserRating, entry: RatingEntry) -> None:
        """Применить поля из RatingEntry к объекту UserRating.

        Единственное место, где задаются поля рейтинга, чтобы при
        добавлении нового поля не дублировать логику.
        """
        rating.place = entry.place
        rating.competition_type = entry.competition_type
        rating.status = entry.status

    # ------------------------------------------------------------------
    # Публичные методы
    # ------------------------------------------------------------------

    async def create_or_update_user_rating(
        self,
        user_id: int,
        leaderboard_id: str,
        place: int,
        competition_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> UserRating:
        """Создать или обновить рейтинг пользователя (одиночный upsert).

        competition_type и status обновляются только если переданы явно (не None).
        """
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
                competition_type=competition_type or "",
                status=status or "",
            )
            self.session.add(user_rating)
            logger.info(
                f"Создана новая запись рейтинга для пользователя {user_id}, "
                f"место: {place}"
            )
        else:
            old_place = user_rating.place
            user_rating.place = place
            if competition_type is not None:
                user_rating.competition_type = competition_type
            if status is not None:
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
        """Обновить позиции абитуриентов в БД по результатам парсера.

        Поддерживает множественные пользователи с одним applicant_id.
        Использует батч-запросы: одним SELECT загружает всех пользователей
        по идентификаторам, вторым — все существующие UserRating для данного
        leaderboard. Затем делает upsert в памяти и один commit.

        Args:
            leaderboard_id: ID рейтинговой таблицы.
            entries: Список записей от парсера.

        Returns:
            {"created": int, "updated": int, "skipped": int}
        """
        stats = {"updated": 0, "created": 0, "skipped": 0}

        if not entries:
            return stats

        identifiers = [e.identifier for e in entries if e.identifier]

        if not identifiers:
            logger.warning("Нет валидных идентификаторов в списке entries")
            stats["skipped"] = len(entries)
            return stats

        # Батч 1: загрузить всех пользователей по списку идентификаторов
        # Может быть несколько пользователей с одним applicant_id!
        users_result = await self.session.execute(
            select(User).where(User.applicant_id.in_(identifiers))
        )
        users_by_identifier: dict[str, list[User]] = {}
        for u in users_result.scalars():
            if u.applicant_id is None:
                continue
            if u.applicant_id not in users_by_identifier:
                users_by_identifier[u.applicant_id] = []
            users_by_identifier[u.applicant_id].append(u)

        # Собрать ID всех найденных пользователей
        known_user_ids = [
            u.user_id for users in users_by_identifier.values() for u in users
        ]

        # Батч 2: загрузить существующие UserRating для этого leaderboard
        if known_user_ids:
            ratings_result = await self.session.execute(
                select(UserRating).where(
                    UserRating.leaderboard_id == leaderboard_id,
                    UserRating.user_id.in_(known_user_ids),
                )
            )
            existing_ratings = {r.user_id: r for r in ratings_result.scalars()}
        else:
            existing_ratings = {}

        # Обработать каждую entry
        for entry in entries:
            if not entry.identifier:
                logger.debug("Пустой идентификатор в entry — пропуск")
                stats["skipped"] += 1
                continue

            users = users_by_identifier.get(entry.identifier)
            if not users:
                logger.debug(
                    f"Абитуриент с идентификатором '{entry.identifier}' "
                    "не найден в БД — пропуск"
                )
                stats["skipped"] += 1
                continue

            # Обновить/создать рейтинг для ВСЕХ пользователей с этим applicant_id
            for user in users:
                user_rating = existing_ratings.get(user.user_id)
                if user_rating is None:
                    user_rating = UserRating(
                        user_id=user.user_id,
                        leaderboard_id=leaderboard_id,
                        place=0,
                    )
                    self.session.add(user_rating)
                    self._apply_entry_fields(user_rating, entry)
                    existing_ratings[user.user_id] = user_rating
                    stats["created"] += 1
                    logger.debug(
                        f"Создана запись рейтинга: user={user.user_id}, "
                        f"place={entry.place}, "
                        f"competition_type='{entry.competition_type}', "
                        f"status='{entry.status}'"
                    )
                else:
                    old_place = user_rating.place
                    self._apply_entry_fields(user_rating, entry)
                    stats["updated"] += 1
                    logger.debug(
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
