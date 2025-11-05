import os

from dotenv import load_dotenv

load_dotenv()

# Настройки БД из файла (не из .env)
DB_ECHO = False
DB_POOL_SIZE = 5
DB_MAX_OVERFLOW = 10


def get_database_url() -> str:
    """Получение URL подключения к базе данных."""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD")
    database = os.getenv("DB_NAME", "abitur_bot")

    if not password:
        raise ValueError("DB_PASSWORD не задан в переменных окружения")

    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"


def get_database_config() -> dict:
    """Настройки для SQLAlchemy engine."""
    return {
        "echo": DB_ECHO,  # Логирование SQL запросов
        "pool_size": DB_POOL_SIZE,  # Размер пула соединений
        "max_overflow": DB_MAX_OVERFLOW,  # Максимум дополнительных соединений
        "pool_pre_ping": True,  # Проверка соединений перед использованием
        "pool_recycle": 3600,  # Пересоздание соединений каждый час
    }


# Экспорт для удобства использования
DATABASE_URL = get_database_url()
DATABASE_CONFIG = get_database_config()
