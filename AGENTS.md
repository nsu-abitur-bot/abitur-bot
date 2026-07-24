# AGENTS.md

Инструкции для AI-агентов, работающих в этом репозитории. Это **канонический источник** правил проекта (на него ссылается CLAUDE.md).

## Проект

**NSU Abitur Bot** — Telegram/MAX-бот для абитуриентов Новосибирского государственного университета. Отвечает на вопросы о поступлении.

Пайплайн ответа (`llm/llm_client.py:ask_local_llm`): расширение аббревиатур → **FAQ-матчер** (при попадании отвечает сразу, минуя RAG/LLM) → **RAG** (Corrective RAG поверх LightRAG, либо обычный LightRAG) → **LLM-генерация** (OpenAI / Gemini) с доступом к **function-calling инструментам** (напр. проходные баллы отвечаются из SQL, а не из RAG).

Три независимых процесса:
- `main.py` — бот (Telegram + MAX)
- `api/main.py` — FastAPI для управления и логов
- `scheduler/main.py` — периодические задачи (конкурсные списки, уведомления)

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

# Линтинг и форматирование (ruff) — только по изменённым файлам, НЕ по всему репо
uv run ruff check .
uv run ruff check . --fix
uv run ruff format <изменённые файлы>

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
  routes/         Эндпоинты по сущностям (faq, abbrev, faculty, admission_score, rag, ...)
  schemas/        Pydantic-схемы запросов/ответов
  services/       Бизнес-логика для роутов
  auth/           JWT-аутентификация (Admin, роли, инвайт-коды)
abbrev/           Раскрытие аббревиатур перед FAQ/RAG (expander.py)
bot/              Telegram и MAX боты, стриминг, уведомления
db/
  postgres/       SQLAlchemy ORM + Alembic миграции
    models.py     Все модели (единственный источник правды)
    services/     Сервисы доступа к данным (относительные импорты внутри)
  redis/
    client.py     Клиент Redis (история чатов, сессии)
  seed/           Сид-данные (faculties.json) + автозаливка при старте
evals/            Оценка качества RAG (evaluator + judge)
faq/              Семантический матчер FAQ (matcher.py, эмбеддинги)
llm/
  providers/      openai.py / gemini.py (+ *_graph_adapters.py для LightRAG)
  tools/          Function-calling инструменты (admission_scores.py, registry.py)
  base.py         BaseLLMProvider, LLMResult, ToolSpec, generate_with_tools
  factory.py      get_llm_provider() — синглтон через LLM_PROVIDER
  llm_client.py   Основная логика: abbrev → FAQ → RAG → LLM(+tools)
  profiles.py     Профили параметров (CHAT, GRAPH, PARSER, VISION, INTENT, TITLE, EMBEDDING)
  preprocessor.py Очистка/структурирование текста перед загрузкой в RAG
parser/           Парсеры сайта НГУ и таблиц
  nsu.py          Страницы факультетов НГУ
  rating.py       Конкурсные списки (abiturient.nsu.ru)
  scores.py       Проходные баллы прошлых лет (страница «Итоги приёма»)
  pdf.py/vision.py Парсинг PDF через Vision LLM
rag/
  graph_memory.py LightRAG: граф знаний + встроенный векторный стор (nano-vectordb, JSON на диске в data/lightrag/). ChromaDB в проекте НЕТ.
  retriever.py    Фасад запросов: query_graph_with_crag / query_graph_with_sources
  crag.py         Corrective RAG: LLM-грейдинг чанков + авторитетная фильтрация по справочнику факультетов (сентенс-левел)
  loader.py       Загрузка документов в LightRAG
scheduler/        Периодические задачи
tests/            Тесты, зеркалирующие структуру кода (БД-тесты — в tests/postgres/)
```

## RAG-пайплайн

- **LightRAG** (`rag/graph_memory.py`) хранит и граф сущностей/связей, и векторный индекс чанков — единым встроенным стором на диске (`data/lightrag/<graph_id>/`, `graph_id="abitur_kb"`). Отдельного ChromaDB нет. Чанкинг внутри LightRAG (`chunk_token_size=400`, overlap 50).
- **CRAG** (`rag/crag.py`) — слой между ретривалом и генерацией: (1) распознаёт факультет/уровень в вопросе по справочнику; (2) сентенс-левел вычищает из чанков предложения про направления ЧУЖИХ факультетов; (3) LLM-грейдит релевантность; (4) при нехватке — одна переформулировка+доретрив. Включается `CRAG_ENABLED` (env) + настройки из таблицы `settings` (веб-админка).
- **Справочник факультетов** (`Faculty`/`Program`, `db/postgres/services/faculty.py`) — авторитетный источник для CRAG. Редактируется через `/api/v1/faculties`. При старте выполняется **мягкая** доливка из `db/seed/faculties.json`: добавляется только недостающее (новые факультеты, алиасы, направления, код у направления без кода), правки из админки не затираются — поэтому новые записи сида подхватываются на каждом деплое. Вручную: `uv run python -m db.seed.load_faculties` (мягко) / `--force` (жёсткая перезапись, также `SEED_FACULTIES_FORCE=1` при старте).
- **Проходные баллы** (`AdmissionScore`, `db/postgres/services/admission_score.py`, `parser/scores.py`, `llm/tools/admission_scores.py`): числовые баллы вынесены в структурное хранилище и отвечаются function-calling инструментом `get_admission_scores` из SQL (не из RAG). Данные заливаются вручную из админки: `POST /admission-scores/preview` → просмотр → `POST /admission-scores/import`.

## Function-calling инструменты

- `ToolSpec` (name, description, JSON-schema параметров) и `BaseLLMProvider.generate_with_tools(...)` — в `llm/base.py`. Провайдеры реализуют нативный tool-loop со стримингом финального ответа.
- Инструменты и диспетчер — в `llm/tools/` (`registry.py`, `default_tool_executor`). Исполнитель открывает свою `AsyncSessionLocal`.
- Новый инструмент: добавить `ToolSpec` + `async execute_*` в `llm/tools/`, зарегистрировать в `registry.py`, передать в `generate_with_tools` в `llm_client.py`.

## База данных

### Модели (db/postgres/models.py)

Все SQLAlchemy-модели в одном файле, наследуют `Base`. Ключевые таблицы:

| Класс | Таблица | Назначение |
|-------|---------|-----------|
| `User` | `user` | Пользователи бота |
| `Message` / `MessageLog` | `message` / `message_logs` | История переписки / детальные логи обработки |
| `Leaderboard` / `UserRating` | `leaderboard` / `user_rating` | Конкурсные списки и позиции пользователей |
| `FaqEntry` / `Abbreviation` | `faq_entry` / `abbreviation` | FAQ и аббревиатуры (грузятся в память при старте) |
| `Topic` | `topic` | Темы для аналитической intent-классификации |
| `Document` | `document` | Метаданные документов в RAG |
| `Faculty` / `Program` | `faculty` / `program` | Справочник факультетов и направлений (для CRAG) |
| `AdmissionScore` | `admission_score` | Проходные баллы прошлых лет (program × year × form) |
| `Settings` | `settings` | Настройки приложения (в т.ч. CRAG) |
| `Admin` / `InviteCode` | `admins` / `invite_codes` | Аутентификация админки |
| `QuestionEmbeddingCache` | `question_embedding_cache` | Кэш эмбеддингов вопросов |

PK-соглашение: `BigInteger autoincrement` для числовых ID, `uuid7()` (String(36)) для строковых.

### Сервисы (db/postgres/services/)

Каждая таблица имеет сервис с async-методами. Сервисы принимают `AsyncSession` в конструктор, используются через `async with AsyncSessionLocal() as session`. **Импорты внутри `services/` — относительные**, во всём остальном коде — абсолютные.

### Миграции

После изменения модели **обязательно** создать миграцию и проверить сгенерированный файл в `db/alembic/versions/` перед коммитом (убрать посторонние autogenerate-операции). Формат имени: `YYYY_MM_DD_HHMM-<rev>_<slug>`.

## Провайдеры LLM

Активные: **OpenAI**, **Gemini**. Выбор через env-переменные:

| Переменная | Назначение | Значения |
|-----------|-----------|---------|
| `LLM_PROVIDER` | Ответы бота | `openai`, `gemini` |
| `LIGHTRAG_LLM_PROVIDER` | LightRAG граф (независимо от бота) | `openai`, `gemini` |
| `PDF_PARSER_PROVIDER` | Парсинг PDF (vision) | `openai`, `gemini` |

Новый провайдер: `providers/<name>.py` + `providers/<name>_graph_adapters.py`, зарегистрировать в `factory.py` и `graph_memory.py`.

## Соглашения по коду

- **Нейминг**: `snake_case` везде (файлы, переменные, функции, модули)
- **Импорты**: абсолютные везде, кроме `db/postgres/services/` (там относительные)
- **Типизация**: аннотации типов обязательны для публичных функций
- **Комментарии**: только когда «почему» неочевидно, не «что делает»
- **Тесты БД**: реальная тестовая база (фикстуры в `tests/postgres/conftest.py`), без mock'ов
- **ruff format**: только по изменённым файлам, не переформатировать весь репо

## Логи

Файлы пишутся в `logs/` (гитигнорен). Настройка через `LOG_LEVEL` и `LOG_DIR` в `.env`.

| Файл | Уровень | Лимит | Назначение |
|------|---------|-------|-----------|
| `logs/abitur_bot.log` | DEBUG+ | 10 MB × 5 | Основной лог всего приложения |
| `logs/rag_llm_detailed.log` | DEBUG+ | 50 MB × 10 | RAG-контекст, LLM-ответы, FAQ-матчи |
| `logs/errors.log` | ERROR+ | 5 MB × 3 | Только ошибки |

Полезные команды для отладки:
```bash
tail -f logs/rag_llm_detailed.log | grep "<session_id>"   # вся цепочка одного запроса
tail -f logs/rag_llm_detailed.log | grep "RAG retrieval"  # что достал RAG
tail -f logs/rag_llm_detailed.log | grep "CRAG"           # решения CRAG-фильтра
tail -f logs/errors.log                                   # текущие ошибки
```

## Линтер и тайп-чекер

**ruff** (`pyproject.toml → [tool.ruff]`): проверяет E, F, I правила. Запускать перед коммитом.

**pyrefly**: статическая проверка типов. Запускать через `uv run pyrefly check`.
