from datetime import UTC, datetime
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Базовый класс для всех моделей."""

    pass


class User(Base):
    """Пользователи Telegram."""

    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    snils_id: Mapped[str] = mapped_column(String(14), unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC).replace(tzinfo=None)
    )

    # Связь с рейтингами
    ratings: Mapped[List["UserRating"]] = relationship(
        "UserRating", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(user_id={self.user_id}, snils_id='{self.snils_id}')>"


class Leaderboard(Base):
    """Рейтинговые таблицы."""

    __tablename__ = "leaderboards"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    url: Mapped[str] = mapped_column(String(500), unique=True)
    title: Mapped[str] = mapped_column(String(200), default="")
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC).replace(tzinfo=None),
        onupdate=lambda: datetime.now(UTC).replace(tzinfo=None),
    )

    # Связь с рейтингами пользователей
    user_ratings: Mapped[List["UserRating"]] = relationship(
        "UserRating", back_populates="leaderboard", cascade="all, delete-orphan"
    )

    # Индексы
    __table_args__ = (
        Index("idx_leaderboards_hash", "content_hash"),
        Index("idx_leaderboards_active", "is_active"),
        Index("idx_leaderboards_updated", "updated_at"),
    )

    def __repr__(self) -> str:
        return f"<Leaderboard(id='{self.id}', url='{self.url}')>"


class UserRating(Base):
    """Рейтинги пользователей в таблицах."""

    __tablename__ = "user_ratings"

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.user_id"), primary_key=True
    )
    leaderboard_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("leaderboards.id"), primary_key=True
    )
    place: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC).replace(tzinfo=None),
        onupdate=lambda: datetime.now(UTC).replace(tzinfo=None),
    )

    # Связи
    user: Mapped["User"] = relationship("User", back_populates="ratings")
    leaderboard: Mapped["Leaderboard"] = relationship(
        "Leaderboard", back_populates="user_ratings"
    )

    def __repr__(self) -> str:
        return f"<UserRating(user_id={self.user_id}, leaderboard_id='{self.leaderboard_id}', place={self.place})>"  # noqa: E501
