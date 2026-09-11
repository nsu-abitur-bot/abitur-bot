# Тесты, линтер и типизация

Открывай, когда пишешь тесты или разбираешься, почему упал CI.

## Тесты

```bash
uv run pytest
uv run pytest tests/<module>/ -v
uv run pytest -k "test_name" -v
```

Запускай после изменений, а не только перед коммитом.

Структура `tests/` зеркалирует структуру кода. БД-тесты — в `tests/postgres/`.

**БД-тесты работают с реальной тестовой базой, без mock'ов** (фикстуры в
`tests/postgres/conftest.py`). База берётся из `TEST_DB_NAME` (по умолчанию
`abitur_test`) и `DB_*` переменных; `conftest` создаёт только таблицы, поэтому
сама база должна существовать — иначе сбор тестов падает.

Redis мокать не нужно: он подменяется `fakeredis`.

**Тест не должен зависеть от файлов вне git.** `abbrev/abbrev_data.yaml` и
`faq/faq_data.yaml` в `.gitignore` — тест, который читает их через реальный
загрузчик, проходит только там, где файл случайно оказался. Нужны данные —
загрузи их в тесте явно (например, `AbbrevExpander.load_items([...])`).

## Линтер и типизация

```bash
uv run ruff check .
uv run ruff check . --fix
uv run ruff format <изменённые файлы>
uv run pyrefly check
```

**ruff** (`pyproject.toml → [tool.ruff]`): правила E, F, I, лимит строки 90
символов. `ruff format` — только по изменённым файлам, не переформатируй весь
репозиторий.

**pyrefly**: статическая проверка типов. Завершается кодом 1 при ошибках, то
есть шаг CI от них падает.

## CI

`.github/workflows/pr-checks.yml`, два job на каждый pull request:

- `lint-and-types` — ruff и pyrefly;
- `tests` — pytest с сервисом PostgreSQL (`postgres:15-alpine`, база
  `abitur_test` создаётся образом).

Job раздельные: тестам нужна база, линтеру нет, и падение теста не должно
прятать ошибку линтера.
