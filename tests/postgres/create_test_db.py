"""
Скрипт для создания тестовой базы данных.
"""

import asyncio
import os
import sys

import asyncpg
from dotenv import load_dotenv

load_dotenv()


async def create_test_database():
    """Создать тестовую БД если её нет."""
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD")
    test_db_name = os.getenv("TEST_DB_NAME", "abitur_test")

    if not db_password:
        print("DB_PASSWORD не задан в .env файле")
        sys.exit(1)

    try:
        # Подключаемся к postgres
        conn = await asyncpg.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database="postgres",
        )

        # Проверяем существование тестовой БД
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", test_db_name
        )

        if not exists:
            await conn.execute(f'CREATE DATABASE "{test_db_name}"')
            print(f"Тестовая БД '{test_db_name}' создана")
        else:
            print(f"Тестовая БД '{test_db_name}' уже существует")

        await conn.close()

    except Exception as e:
        print(f"Ошибка при создании тестовой БД: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(create_test_database())
