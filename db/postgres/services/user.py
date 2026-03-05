import logging
from datetime import UTC, datetime, timedelta
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import User

logger = logging.getLogger(__name__)


class UserService:
    """Сервис для работы с пользователями."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_user(
        self, user_id: int, applicant_id: Optional[str] = None
    ) -> Optional[User]:
        """Создать нового пользователя."""
        try:
            user = User(user_id=user_id, applicant_id=applicant_id)
            self.session.add(user)
            await self.session.commit()
            await self.session.refresh(user)
            logger.info(f"Пользователь {user_id} создан")
            return user
        except IntegrityError as e:
            await self.session.rollback()
            logger.warning(
                f"Не удалось создать пользователя {user_id} из-за "
                f"нарушения ограничения БД: {e}"
            )
            return None
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка создания пользователя {user_id}: {e}")
            return None

    async def get_user(self, user_id: int) -> Optional[User]:
        """Получить пользователя по ID."""
        try:
            result = await self.session.execute(
                select(User).where(User.user_id == user_id)
            )
            user = result.scalar_one_or_none()
            if user:
                logger.debug(f"Пользователь {user_id} найден")
            else:
                logger.debug(f"Пользователь {user_id} не найден")
            return user
        except Exception as e:
            logger.error(f"Ошибка получения пользователя {user_id}: {e}")
            return None

    async def get_users_by_applicant_id(self, applicant_id: str) -> List[User]:
        """Получить всех пользователей по идентификатору абитуриента.

        Возвращает список пользователей, т.к. несколько пользователей
        могут быть подписаны на один applicant_id.
        """
        try:
            result = await self.session.execute(
                select(User).where(User.applicant_id == applicant_id)
            )
            users = list(result.scalars().all())
            logger.debug(
                f"Найдено {len(users)} пользователей с applicant_id {applicant_id}"
            )
            return users
        except Exception as e:
            logger.error(
                f"Ошибка получения пользователей по applicant_id {applicant_id}: {e}"
            )
            return []

    async def get_all_users(self) -> List[User]:
        """Получить всех пользователей."""
        try:
            result = await self.session.execute(select(User).order_by(User.created_at))
            users = result.scalars().all()
            logger.debug(f"Получено {len(users)} пользователей")
            return list(users)
        except Exception as e:
            logger.error(f"Ошибка получения всех пользователей: {e}")
            return []

    async def user_exists(self, user_id: int) -> bool:
        """Проверить существование пользователя."""
        try:
            result = await self.session.execute(
                select(User.user_id).where(User.user_id == user_id)
            )
            exists = result.scalar() is not None
            logger.debug(
                f"Пользователь {user_id} {'существует' if exists else 'не существует'}"
            )
            return exists
        except Exception as e:
            logger.error(f"Ошибка проверки существования пользователя {user_id}: {e}")
            return False

    async def get_user_count(self) -> int:
        """Получить количество пользователей."""
        try:
            result = await self.session.execute(select(func.count(User.user_id)))
            count = result.scalar() or 0
            logger.debug(f"Всего пользователей: {count}")
            return count
        except Exception as e:
            logger.error(f"Ошибка подсчета пользователей: {e}")
            return 0

    async def get_user_count_stats(self) -> dict[str, int]:
        """Получить количество пользователей за разные периоды."""
        try:
            now = datetime.now(UTC).replace(tzinfo=None)
            day_ago = now - timedelta(days=1)
            week_ago = now - timedelta(weeks=1)
            month_ago = now - timedelta(days=30)
            year_ago = now - timedelta(days=365)

            result = await self.session.execute(
                select(
                    func.count(User.user_id)
                    .filter(User.created_at >= day_ago)
                    .label("day"),
                    func.count(User.user_id)
                    .filter(User.created_at >= week_ago)
                    .label("week"),
                    func.count(User.user_id)
                    .filter(User.created_at >= month_ago)
                    .label("month"),
                    func.count(User.user_id)
                    .filter(User.created_at >= year_ago)
                    .label("year"),
                    func.count(User.user_id).label("all_time"),
                )
            )

            row = result.one()
            return {
                "day": int(row.day or 0),
                "week": int(row.week or 0),
                "month": int(row.month or 0),
                "year": int(row.year or 0),
                "all_time": int(row.all_time or 0),
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики пользователей: {e}")
            raise

    async def update_applicant_id(
        self, user_id: int, new_applicant_id: Optional[str]
    ) -> bool:
        """Обновить идентификатор абитуриента."""
        try:
            result = await self.session.execute(
                select(User).where(User.user_id == user_id)
            )
            user = result.scalar_one_or_none()

            if not user:
                logger.warning(
                    f"Пользователь {user_id} не найден для обновления applicant_id"
                )
                return False

            user.applicant_id = new_applicant_id
            await self.session.commit()
            logger.info(
                f"applicant_id пользователя {user_id} обновлен на {new_applicant_id}"
            )
            return True
        except IntegrityError as e:
            await self.session.rollback()
            logger.warning(
                f"Ошибка обновления applicant_id для пользователя {user_id}: {e}"
            )
            return False
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка обновления applicant_id пользователя {user_id}: {e}")
            return False

    async def delete_user(self, user_id: int) -> bool:
        """Удалить пользователя."""
        try:
            result = await self.session.execute(
                select(User).where(User.user_id == user_id)
            )
            user = result.scalar_one_or_none()

            if not user:
                logger.warning(f"Пользователь {user_id} не найден для удаления")
                return False

            await self.session.delete(user)
            await self.session.commit()
            logger.info(f"Пользователь {user_id} удален")
            return True
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка удаления пользователя {user_id}: {e}")
            return False
