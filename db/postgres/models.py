import enum
from datetime import UTC, datetime
from typing import List, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from uuid6 import uuid7


class AdminRole(str, enum.Enum):
    superadmin = "superadmin"
    admin = "admin"
    viewer = "viewer"


class EducationLevel(str, enum.Enum):
    """Уровень образования направления подготовки."""

    bachelor = "bachelor"
    specialist = "specialist"
    master = "master"


# Допустимые значения уровня образования (для валидации сидов/API).
EDUCATION_LEVELS: frozenset[str] = frozenset(level.value for level in EducationLevel)


class AdmissionForm(str, enum.Enum):
    """Форма обучения для проходных баллов."""

    budget = "budget"
    paid = "paid"


# Допустимые значения формы обучения (для валидации сидов/API).
ADMISSION_FORMS: frozenset[str] = frozenset(f.value for f in AdmissionForm)


def timestamp():
    """Генерация текущего времени UTC без timezone info."""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Базовый класс для всех моделей."""

    pass


class User(Base):
    """Пользователи бота с внутренним идентификатором."""

    __tablename__ = "user"

    # Внутренний идентификатор пользователя (не внешний ID мессенджера)
    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    telegram_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    max_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    applicant_id: Mapped[Optional[str]] = mapped_column(String(7), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=timestamp)

    ratings: Mapped[List["UserRating"]] = relationship(
        "UserRating", back_populates="user", cascade="all, delete-orphan"
    )
    messages: Mapped[List["Message"]] = relationship(
        "Message", back_populates="user", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ux_user_telegram_id", "telegram_id", unique=True),
        Index("ux_user_max_id", "max_id", unique=True),
        Index("idx_user_applicant_id", "applicant_id"),
        Index("idx_user_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<User(user_id={self.user_id}, telegram_id={self.telegram_id}, "
            f"max_id='{self.max_id}', applicant_id='{self.applicant_id}')>"
        )


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


class MessageLog(Base):
    """Детальные логи обработки сообщений."""

    __tablename__ = "message_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    message_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # 'user_input', 'rag_context', 'llm_response', 'faq_match'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )  # для источников, длины ответа и т.д.
    topic_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("topic.id"), nullable=True
    )
    tokens_used: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # суммарное количество токенов на генерацию (для message_type='llm_response')
    created_at: Mapped[datetime] = mapped_column(DateTime, default=timestamp)

    topic: Mapped[Optional["Topic"]] = relationship("Topic")

    __table_args__ = (
        Index("ix_message_logs_user_id", "user_id"),
        Index("ix_message_logs_session_id", "session_id"),
        Index("ix_message_logs_created_at", "created_at"),
        Index("ix_message_logs_message_type", "message_type"),
        Index("ix_message_logs_topic_id", "topic_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<MessageLog(id={self.id}, user_id={self.user_id}, "
            f"message_type='{self.message_type}', session_id='{self.session_id}')>"
        )


class QuestionEmbeddingCache(Base):
    """Кеш embeddings для семантической группировки пользовательских вопросов."""

    __tablename__ = "question_embedding_cache"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid7())
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    question_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_question: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=timestamp)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=timestamp, onupdate=timestamp
    )

    __table_args__ = (
        UniqueConstraint(
            "provider",
            "model",
            "question_hash",
            name="uq_question_embedding_cache_provider_model_hash",
        ),
        Index("ix_question_embedding_cache_lookup", "provider", "model", "question_hash"),
    )

    def __repr__(self) -> str:
        return (
            f"<QuestionEmbeddingCache(id='{self.id}', provider='{self.provider}', "
            f"model='{self.model}')>"
        )


class FeedbackReport(Base):
    """Обратная связь пользователей о некорректных ответах бота."""

    __tablename__ = "feedback_report"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    channel: Mapped[str] = mapped_column(String(50), nullable=False)
    comment: Mapped[str] = mapped_column(Text, nullable=False)
    question: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bot_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    logs_snapshot: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=timestamp)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_feedback_report_user_id", "user_id"),
        Index("ix_feedback_report_session_id", "session_id"),
        Index("ix_feedback_report_status", "status"),
        Index("ix_feedback_report_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<FeedbackReport(id={self.id}, user_id={self.user_id}, "
            f"status='{self.status}', session_id='{self.session_id}')>"
        )


class Settings(Base):
    """Настройки приложения."""

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=timestamp, onupdate=timestamp
    )

    def __repr__(self) -> str:
        return f"<Settings(key='{self.key}', value='{self.value}')>"


class Document(Base):
    """Документы, проиндексированные в RAG."""

    __tablename__ = "document"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid7())
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    graph_id: Mapped[str] = mapped_column(String(100), nullable=False)
    rag_doc_id: Mapped[str] = mapped_column(String(500), nullable=False)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    content_length: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="проиндексирован"
    )
    last_indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_checked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_checked_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    last_check_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=timestamp)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=timestamp, onupdate=timestamp
    )

    __table_args__ = (
        UniqueConstraint("graph_id", "rag_doc_id", name="uq_document_graph_rag_doc_id"),
        Index("ix_document_graph_id", "graph_id"),
        Index("ix_document_source_url", "source_url"),
        Index("ix_document_content_hash", "content_hash"),
        Index("ix_document_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<Document(id='{self.id}', title='{self.title}', status='{self.status}')>"


class FaqEntry(Base):
    """FAQ вопросы и ответы."""

    __tablename__ = "faq_entry"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid7())
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    aliases: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=timestamp)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=timestamp, onupdate=timestamp
    )

    def __repr__(self) -> str:
        return f"<FaqEntry(id='{self.id}', question='{self.question[:50]}')>"


class Abbreviation(Base):
    """Словарь аббревиатур."""

    __tablename__ = "abbreviation"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid7())
    )
    short: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    full: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=timestamp)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=timestamp, onupdate=timestamp
    )

    __table_args__ = (Index("idx_abbreviation_short", "short"),)

    def __repr__(self) -> str:
        return f"<Abbreviation(id='{self.id}', short='{self.short}')>"


class Topic(Base):
    """Справочник тем сообщений."""

    __tablename__ = "topic"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=timestamp)

    def __repr__(self) -> str:
        return f"<Topic(id={self.id}, label='{self.label}')>"


class Admin(Base):
    """Администраторы системы (не путать с User — это пользователи бота)."""

    __tablename__ = "admins"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid7())
    )
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False, default=AdminRole.admin)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=timestamp)
    created_by_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("admins.id"), nullable=True
    )

    created_by: Mapped[Optional["Admin"]] = relationship(
        "Admin",
        remote_side=[id],
        foreign_keys=[created_by_id],
    )
    invite_codes: Mapped[List["InviteCode"]] = relationship(
        "InviteCode",
        back_populates="creator",
        foreign_keys="InviteCode.created_by_id",
    )

    __table_args__ = (Index("idx_admin_username", "username"),)

    def __repr__(self) -> str:
        return f"<Admin(id='{self.id}', username='{self.username}', role='{self.role}')>"


class InviteCode(Base):
    """Одноразовые коды для регистрации новых администраторов."""

    __tablename__ = "invite_codes"

    code: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_by_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("admins.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(50), nullable=False, default=AdminRole.admin)
    used_by_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("admins.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=timestamp)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    creator: Mapped["Admin"] = relationship(
        "Admin",
        back_populates="invite_codes",
        foreign_keys=[created_by_id],
    )
    used_by: Mapped[Optional["Admin"]] = relationship(
        "Admin",
        foreign_keys=[used_by_id],
    )

    def __repr__(self) -> str:
        used = self.used_at is not None
        return f"<InviteCode(code='{self.code}', role='{self.role}', used={used})>"


class Faculty(Base):
    """Факультет/институт НГУ.

    Авторитетный справочник для CRAG-фильтрации: связывает факультет с его
    направлениями подготовки. Поле ``aliases`` хранит альтернативные названия и
    аббревиатуры (например, "ФИТ", "Факультет информационных технологий").
    """

    __tablename__ = "faculty"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid7())
    )
    name: Mapped[str] = mapped_column(String(300), nullable=False, unique=True)
    # Список строк: аббревиатуры и альтернативные названия факультета.
    aliases: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=timestamp)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=timestamp, onupdate=timestamp
    )

    programs: Mapped[List["Program"]] = relationship(
        "Program",
        back_populates="faculty",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_faculty_name", "name"),
        Index("ix_faculty_is_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<Faculty(id='{self.id}', name='{self.name}')>"


class Program(Base):
    """Направление подготовки (образовательная программа).

    Одно и то же название может существовать на разных уровнях образования
    (бакалавриат / специалитет / магистратура), поэтому уникальность задаётся
    тройкой (faculty_id, name, level).
    """

    __tablename__ = "program"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid7())
    )
    faculty_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("faculty.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    # Код направления (ФГОС), например "09.03.01". Необязателен.
    code: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    # Уровень образования: bachelor / specialist / master.
    level: Mapped[str] = mapped_column(
        String(20), nullable=False, default=EducationLevel.bachelor.value
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=timestamp)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=timestamp, onupdate=timestamp
    )

    faculty: Mapped["Faculty"] = relationship("Faculty", back_populates="programs")
    admission_scores: Mapped[List["AdmissionScore"]] = relationship(
        "AdmissionScore",
        back_populates="program",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "faculty_id", "name", "level", name="uq_program_faculty_name_level"
        ),
        Index("ix_program_faculty_id", "faculty_id"),
        Index("ix_program_level", "level"),
        Index("ix_program_name", "name"),
    )

    def __repr__(self) -> str:
        return (
            f"<Program(id='{self.id}', name='{self.name}', "
            f"level='{self.level}', faculty_id='{self.faculty_id}')>"
        )


class AdmissionScore(Base):
    """Проходные баллы направления подготовки за прошлые годы.

    Структурированное хранилище (одна строка на тройку program/year/form),
    чтобы отвечать на числовые вопросы из SQL, а не из RAG. Формы обучения —
    только бюджет и платное (``budget`` / ``paid``).
    """

    __tablename__ = "admission_score"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid7())
    )
    program_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("program.id", ondelete="CASCADE"), nullable=False
    )
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    # Форма обучения: budget / paid.
    form: Mapped[str] = mapped_column(String(20), nullable=False)
    passing_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    average_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=timestamp)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=timestamp, onupdate=timestamp
    )

    program: Mapped["Program"] = relationship(
        "Program", back_populates="admission_scores"
    )

    __table_args__ = (
        UniqueConstraint(
            "program_id",
            "year",
            "form",
            name="uq_admission_score_program_year_form",
        ),
        Index("ix_admission_score_program", "program_id"),
        Index("ix_admission_score_year", "year"),
    )

    def __repr__(self) -> str:
        return (
            f"<AdmissionScore(id='{self.id}', program_id='{self.program_id}', "
            f"year={self.year}, form='{self.form}', "
            f"passing_score={self.passing_score})>"
        )
