# База данных

Открывай, когда меняешь модели, пишешь миграцию или ищешь, где лежат данные.

## Модели (`db/postgres/models.py`)

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

PK-соглашение: `BigInteger autoincrement` для числовых ID, `uuid7()`
(String(36)) для строковых.

## Сервисы (`db/postgres/services/`)

Каждая таблица имеет сервис с async-методами. Сервисы принимают `AsyncSession` в
конструктор, используются через `async with AsyncSessionLocal() as session`.

**Импорты внутри `services/` — относительные**, во всём остальном коде —
абсолютные.

## Миграции

После изменения модели **обязательно** создать миграцию и проверить
сгенерированный файл в `db/alembic/versions/` перед коммитом: убрать посторонние
autogenerate-операции. Формат имени: `YYYY_MM_DD_HHMM-<rev>_<slug>`.

```bash
uv run alembic revision --autogenerate -m "описание изменений"
uv run alembic upgrade head
uv run alembic downgrade -1
uv run alembic current
```

## Redis

`db/redis/client.py` — история чатов и сессии. Учти: история не только
сохраняется, но и **читается обратно при сборке промпта**, поэтому без Redis
пайплайн ответа не деградирует, а падает целиком.
