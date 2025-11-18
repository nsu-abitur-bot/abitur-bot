import os
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()


def get_database_url() -> str:
    """Получение URL подключения к базе данных."""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD")
    database = os.getenv("DB_NAME", "abitur")

    if not password:
        raise ValueError("DB_PASSWORD не задан в переменных окружения")

    # URL-кодируем пароль для безопасности
    encoded_password = quote_plus(password)

    return f"postgresql+asyncpg://{user}:{encoded_password}@{host}:{port}/{database}"


def get_database_config() -> dict:
    """Настройки для SQLAlchemy engine."""
    echo = os.getenv("DB_ECHO", "False").lower() in ("true", "1", "yes", "on")
    pool_size = int(os.getenv("DB_POOL_SIZE", "5"))
    max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))

    return {
        "echo": echo,
        "pool_size": pool_size,
        "max_overflow": max_overflow,
        "pool_pre_ping": True,
        "pool_recycle": 3600,
    }


# Экспорт для удобства использования
DATABASE_URL = get_database_url()
DATABASE_CONFIG = get_database_config()
