from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

# Загружаем переменные окружения
load_dotenv()

# Импортируем модели и конфигурацию БД
from db.postgres.config import get_database_url  # noqa: E402
from db.postgres.models import Base  # noqa: E402

# Конфигурация Alembic
config = context.config

# Получаем асинхронный URL и конвертируем в синхронный для миграций
async_url = get_database_url()
# Заменяем postgresql+asyncpg на postgresql+psycopg2
sync_url = async_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")

# Устанавливаем синхронный URL для миграций
config.set_main_option("sqlalchemy.url", sync_url)

# Настройка логирования
if context.config.attributes.get("configure_logger", True):
    if config.config_file_name is not None:
        fileConfig(config.config_file_name, disable_existing_loggers=False)

# Метаданные наших моделей
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Запуск миграций в offline режиме."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Запуск миграций в online режиме."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
