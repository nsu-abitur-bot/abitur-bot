import logging
from datetime import UTC, datetime, timedelta
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import User

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UserService:
    """Сервис для работы с пользователями."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_user(
        self, user_id: int, snils: Optional[str] = None
    ) -> Optional[User]:
        """Создать нового пользователя."""
        try:
            user = User(user_id=user_id, snils_id=snils)
            self.session.add(user)
            await self.session.commit()
            await self.session.refresh(user)
            logger.info(f"Пользователь {user_id} создан")
            return user
        except IntegrityError as e:
            await self.session.rollback()
            logger.warning(
                f"Пользователь {user_id} уже существует или SNILS {snils} занят: {e}"
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

    async def get_user_by_snils(self, snils: str) -> Optional[User]:
        """Получить пользователя по SNILS."""
        try:
            result = await self.session.execute(
                select(User).where(User.snils_id == snils)
            )
            user = result.scalar_one_or_none()
            if user:
                logger.debug(f"Пользователь со SNILS {snils} найден")
            else:
                logger.debug(f"Пользователь со SNILS {snils} не найден")
            return user
        except Exception as e:
            logger.error(f"Ошибка получения пользователя по SNILS {snils}: {e}")
            return None

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
            return {
                "day": 0,
                "week": 0,
                "month": 0,
                "year": 0,
                "all_time": 0,
            }

    async def update_snils(self, user_id: int, new_snils: Optional[str]) -> bool:
        """Обновить SNILS пользователя."""
        try:
            result = await self.session.execute(
                select(User).where(User.user_id == user_id)
            )
            user = result.scalar_one_or_none()

            if not user:
                logger.warning(f"Пользователь {user_id} не найден для обновления SNILS")
                return False

            user.snils_id = new_snils
            await self.session.commit()
            logger.info(f"SNILS пользователя {user_id} обновлен на {new_snils}")
            return True
        except IntegrityError:
            await self.session.rollback()
            logger.warning(f"SNILS {new_snils} уже используется")
            return False
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка обновления SNILS пользователя {user_id}: {e}")
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
