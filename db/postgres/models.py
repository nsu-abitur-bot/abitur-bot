from datetime import UTC, datetime
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from uuid6 import uuid7


def timestamp():
    """Генерация текущего времени UTC без timezone info."""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Базовый класс для всех моделей."""

    pass


class User(Base):
    """Пользователи Telegram."""

    __tablename__ = "user"

    # Это user_id из телеграмма
    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    applicant_id: Mapped[Optional[str]] = mapped_column(String(7), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=timestamp)

    ratings: Mapped[List["UserRating"]] = relationship(
        "UserRating", back_populates="user", cascade="all, delete-orphan"
    )
    messages: Mapped[List["Message"]] = relationship(
        "Message", back_populates="user", cascade="all, delete-orphan"
    )
    feedbacks: Mapped[List["Feedback"]] = relationship(
        "Feedback", back_populates="user", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_user_applicant_id", "applicant_id"),
        Index("idx_user_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<User(user_id={self.user_id}, applicant_id='{self.applicant_id}')>"


class Leaderboard(Base):
    """Рейтинговые таблицы."""

    __tablename__ = "leaderboard"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid7())
    )
    url: Mapped[str] = mapped_column(String(500), unique=True)
    direction: Mapped[str] = mapped_column(String(200), default="")
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=timestamp,
        onupdate=timestamp,
    )

    user_ratings: Mapped[List["UserRating"]] = relationship(
        "UserRating", back_populates="leaderboard", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_leaderboards_hash", "content_hash"),
        Index("idx_leaderboards_active", "is_active"),
        Index("idx_leaderboards_updated", "updated_at"),
    )

    def __repr__(self) -> str:
        return f"<Leaderboard(id='{self.id}', url='{self.url}')>"


class UserRating(Base):
    """Рейтинги пользователей в таблицах."""

    __tablename__ = "user_rating"

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("user.user_id"), primary_key=True
    )
    leaderboard_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("leaderboard.id"), primary_key=True
    )
    place: Mapped[int] = mapped_column(Integer)
    competition_type: Mapped[str] = mapped_column(
        String(200), default="", server_default=""
    )
    status: Mapped[str] = mapped_column(String(100), default="", server_default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=timestamp,
        onupdate=timestamp,
    )

    user: Mapped["User"] = relationship("User", back_populates="ratings")
    leaderboard: Mapped["Leaderboard"] = relationship(
        "Leaderboard", back_populates="user_ratings"
    )

    def __repr__(self) -> str:
        return (
            f"<UserRating(user_id={self.user_id}, "
            f"leaderboard_id='{self.leaderboard_id}', "
            f"place={self.place}, competition_type='{self.competition_type}', "
            f"status='{self.status}')>"
        )


class Message(Base):
    """Сообщения пользователей и ответы бота."""

    __tablename__ = "message"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid7())
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("user.user_id"), nullable=False
    )
    session_id: Mapped[str] = mapped_column(String(100), nullable=False)
    user_text: Mapped[str] = mapped_column(Text, nullable=False)
    bot_response: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=timestamp)

    user: Mapped["User"] = relationship("User", back_populates="messages")

    __table_args__ = (
        Index("idx_message_user_id", "user_id"),
        Index("idx_message_session_id", "session_id"),
        Index("idx_message_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Message(id='{self.id}', user_id={self.user_id}, "
            f"session_id='{self.session_id}')>"
        )


class Feedback(Base):
    """Обратная связь от пользователей."""

    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid7())
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("user.user_id"), nullable=False
    )
    session_id: Mapped[str] = mapped_column(String(100), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=timestamp)

    user: Mapped["User"] = relationship("User", back_populates="feedbacks")

    __table_args__ = (
        Index("idx_feedback_user_id", "user_id"),
        Index("idx_feedback_session_id", "session_id"),
        Index("idx_feedback_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Feedback(id='{self.id}', user_id={self.user_id}, "
            f"session_id='{self.session_id}')>"
        )
