# CLAUDE.md

Инструкции для Claude Code при работе в этом репозитории.

## Проект

Telegram/MAX-бот для абитуриентов НГУ. Отвечает на вопросы через RAG (LightRAG + ChromaDB), FAQ-матчер и LLM (OpenAI или Gemini). Состоит из трёх отдельно запускаемых процессов: бот, API, планировщик.

## Команды

Все команды запускаются через `uv run`. Никогда не используй `python` напрямую.

```bash
# Запуск
uv run python main.py                              # бот (основной процесс)
uv run uvicorn api.main:app --reload               # API (разработка)
uv run python -m scheduler.main                    # планировщик

# Тесты
uv run pytest                                      # все тесты
uv run pytest tests/llm/ -v                        # конкретный модуль
uv run pytest -k "test_name"                       # конкретный тест

# Линтер / форматтер
uv run ruff check .                                # проверка
uv run ruff check . --fix                          # автоисправление
uv run ruff format .                               # форматирование

# Тайп-чекер
uv run pyrefly check                               # проверка типов

# Миграции БД
uv run alembic revision --autogenerate -m "описание"   # создать миграцию
uv run alembic upgrade head                            # применить миграции
uv run alembic downgrade -1                            # откатить последнюю
```

## Архитектура

```
api/          FastAPI: routes/, schemas/, services/
abbrev/       Раскрытие аббревиатур (expander.py)
bot/          Telegram + MAX боты
baza/         Markdown-файлы с данными по факультетам (источник для RAG)
db/
  postgres/   SQLAlchemy модели, сервисы, Alembic миграции
  redis/      Redis-клиент для истории чатов
evals/        Оценка качества RAG-ответов
faq/          FAQ-матчер на эмбеддингах (matcher.py)
llm/
  providers/  Реализации провайдеров (openai.py, gemini.py)
              + graph-адаптеры для LightRAG
  factory.py  Синглтон провайдера через LLM_PROVIDER
  llm_client.py  Основная логика обработки сообщений
parser/       Парсеры NSU-сайта и рейтинговых таблиц
rag/          LightRAG graph memory + ChromaDB retriever
scheduler/    Периодические задачи (обновление рейтингов)
tests/        Зеркалирует структуру кода
```

## Провайдеры LLM

Поддерживаются только **OpenAI** и **Gemini**. Выбор через `.env`:

```
LLM_PROVIDER=gemini              # для ответов бота
LIGHTRAG_LLM_PROVIDER=gemini     # для LightRAG графа
PDF_PARSER_PROVIDER=gemini       # для парсинга PDF
```

## Соглашения

- Всё в `snake_case`, без исключений
- Относительные импорты только внутри `db/postgres/services/` — везде остальное абсолютные
- Тесты для БД используют реальную тестовую БД, не mock'и
- Новые провайдеры LLM: добавить `providers/<name>.py` + `providers/<name>_graph_adapters.py`, зарегистрировать в `factory.py` и `graph_memory.py`

## База данных

Модели живут в `db/postgres/models.py`, наследуют от `Base`. Сервисы (бизнес-логика над БД) — в `db/postgres/services/`.

После изменения модели всегда создавай миграцию:
```bash
uv run alembic revision --autogenerate -m "краткое описание"
```
Проверь сгенерированный файл в `db/alembic/versions/` перед применением.
