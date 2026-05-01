# AGENTS.md

Инструкции для AI-агентов, работающих в этом репозитории.

## Проект

**NSU Abitur Bot** — Telegram/MAX-бот для абитуриентов Новосибирского государственного университета. Отвечает на вопросы о поступлении через трёхуровневый пайплайн: FAQ-матчер → LightRAG граф → LLM (OpenAI / Gemini).

Три независимых процесса:
- `main.py` — бот (Telegram + MAX)
- `api/main.py` — FastAPI для управления и логов
- `scheduler/main.py` — периодические задачи (рейтинги, уведомления)

## Окружение

Менеджер зависимостей: **uv**. Никогда не используй `pip` или голый `python`.

```bash
uv run <command>     # запуск любой команды в venv проекта
uv add <package>     # добавить зависимость
uv sync              # синхронизировать окружение после изменения pyproject.toml
```

## Основные команды

```bash
# Запуск
uv run python main.py
uv run uvicorn api.main:app --reload --port 8000
uv run python -m scheduler.main

# Тесты (всегда запускай после изменений)
uv run pytest
uv run pytest tests/<module>/ -v
uv run pytest -k "test_name" -v

# Линтинг и форматирование (ruff)
uv run ruff check .
uv run ruff check . --fix
uv run ruff format .

# Типизация (pyrefly)
uv run pyrefly check

# Миграции (alembic)
uv run alembic revision --autogenerate -m "описание изменений"
uv run alembic upgrade head
uv run alembic downgrade -1
uv run alembic current
```

## Структура модулей

```
api/              HTTP API (FastAPI)
  routes/         Эндпоинты по сущностям
  schemas/        Pydantic-схемы запросов/ответов
  services/       Бизнес-логика для роутов
abbrev/           Раскрытие аббревиатур перед RAG-запросом
bot/              Telegram и MAX боты
db/
  postgres/       SQLAlchemy ORM + Alembic миграции
    models.py     Все модели (единственный источник правды)
    services/     Сервисы доступа к данным (UserService и т.д.)
    config.py     DATABASE_URL из env
  redis/
    client.py     Клиент Redis (история чатов, сессии)
evals/            Оценка качества RAG (evaluator + judge)
faq/              Семантический матчер FAQ-вопросов
llm/
  providers/      Реализации провайдеров
    openai.py              OpenAIProvider
    gemini.py              GeminiProvider
    openai_graph_adapters.py   LightRAG-адаптеры для OpenAI
    gemini_graph_adapters.py   LightRAG-адаптеры для Gemini
  base.py         BaseLLMProvider (абстракция)
  factory.py      get_llm_provider() — синглтон через LLM_PROVIDER
  llm_client.py   Основная логика: FAQ → RAG → LLM
  profiles.py     Профили параметров (CHAT, INTENT, EVAL)
  vision_parser.py  Парсинг PDF через Vision
parser/           Парсеры сайта НГУ и рейтинговых таблиц
rag/
  graph_memory.py   LightRAG (граф знаний)
  retriever.py      ChromaDB (векторный поиск)
  loader.py         Загрузка документов в RAG
scheduler/        Периодические задачи
tests/            Тесты, зеркалирующие структуру кода
```

## База данных

### Модели (db/postgres/models.py)

Все SQLAlchemy-модели в одном файле, наследуют `Base`. Текущие таблицы:

| Класс | Таблица | Назначение |
|-------|---------|-----------|
| `User` | `user` | Пользователи бота |
| `Leaderboard` | `leaderboard` | Рейтинговые таблицы НГУ |
| `UserRating` | `user_rating` | Позиция пользователя в рейтинге |
| `Message` | `message` | История переписки |
| `MessageLog` | `message_logs` | Детальные логи обработки |
| `Settings` | `settings` | Настройки приложения |

PK-соглашение: `BigInteger autoincrement` для числовых ID, `uuid7()` (String(36)) для строковых.

### Сервисы (db/postgres/services/)

Каждая таблица имеет сервис с async-методами. Сервисы принимают `AsyncSession` в конструктор. Используются через `async with AsyncSessionLocal() as session`.

### Миграции

После изменения модели **обязательно** создать миграцию и проверить сгенерированный файл в `db/alembic/versions/` перед коммитом. Формат имени: `YYYY_MM_DD_HHMM-<rev>_<slug>`.

## Провайдеры LLM

Активные: **OpenAI**, **Gemini**. Выбор через env-переменные:

| Переменная | Назначение | Значения |
|-----------|-----------|---------|
| `LLM_PROVIDER` | Ответы бота | `openai`, `gemini` |
| `LIGHTRAG_LLM_PROVIDER` | LightRAG граф | `openai`, `gemini` |
| `PDF_PARSER_PROVIDER` | Парсинг PDF | `openai`, `gemini` |

## Соглашения по коду

- **Нейминг**: `snake_case` везде (файлы, переменные, функции, модули)
- **Импорты**: абсолютные везде, кроме `db/postgres/services/` (там относительные)
- **Типизация**: аннотации типов обязательны для публичных функций
- **Комментарии**: только когда «почему» неочевидно, не «что делает»
- **Тесты БД**: реальная тестовая база (`TEST_DB_NAME`), без mock'ов

## Линтер и тайп-чекер

**ruff** (`pyproject.toml → [tool.ruff]`): проверяет E, F, I правила. Запускать перед коммитом.

**pyrefly**: статическая проверка типов. Конфигурация — в `pyproject.toml` (секция `[tool.pyrefly]` если нужна).

Оба инструмента запускать через `uv run`.
