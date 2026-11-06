# База данных

Открывай, когда меняешь модели, пишешь миграцию или ищешь, где лежат данные.

## Модели (`db/postgres/models.py`)

Все SQLAlchemy-модели в одном файле, наследуют `Base`. Ключевые таблицы:

| Класс | Таблица | Назначение |
|-------|---------|-----------|
| `User` | `user` | Пользователи бота |
| `Message` / `MessageLog` | `message` / `message_logs` | История переписки / детальные логи обработки |
| `Leaderboard` / `UserRating` | `leaderboard` / `user_rating` | Конкурсные списки и позиции пользователей |
| `NotificationOutbox` | `notification_outbox` | Очередь уведомлений об изменении позиции: пишется в одном коммите с местами, рассылается планировщиком |
| `FaqEntry` / `Abbreviation` | `faq_entry` / `abbreviation` | FAQ и аббревиатуры (грузятся в память при старте) |
| `Topic` | `topic` | Темы для аналитической intent-классификации |
| `Document` | `document` | Метаданные документов в RAG |
| `ContentProposal` | `content_proposal` | Очередь материалов модели и парсеров до проверки редактором |
| `Faculty` / `Program` | `faculty` / `program` | Справочник факультетов и направлений (для CRAG) |
| `AdmissionScore` | `admission_score` | Проходные баллы прошлых лет (program × year × form) |
| `Settings` | `settings` | Настройки приложения, CRAG и зашифрованный снимок подключений `instance_settings_v1` |
| `Admin` / `InviteCode` | `admins` / `invite_codes` | Аутентификация админки |
| `QuestionEmbeddingCache` | `question_embedding_cache` | Кэш эмбеддингов вопросов |

PK-соглашение: `BigInteger autoincrement` для числовых ID, `uuid7()`
(String(36)) для строковых.

`AdmissionScore.passing_score` и `average_score` хранят дробные значения:
для СПО это баллы аттестата (например, 4,43). Не округлять их до целого.
Откат миграции дробных проходных баллов запрещён, пока в таблице есть
нецелые значения, чтобы не потерять точность.

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

### Уникальность URL документов

Индекс `ix_document_graph_source_url_active` допускает один активный документ
на `(graph_id, source_url)`. `NULL`, пустая строка и статусы `deleted`/`удалён`
исключены из уникальности. При обновлении метаданных `source_url=None` означает
оставить ссылку, `source_url=''` — очистить её.

Миграция `c3a4e7b92f10` согласует существующие дубли перед созданием индекса:
активной остаётся запись с максимальными `updated_at`, затем `id` (порядок
`get_by_source_url`), остальные получают статус `удалён`. Все остальные поля,
включая URL, `rag_doc_id` и временные метки, сохраняются; содержимое RAG не
удаляется. На время согласования и создания индекса блокируются записи в
`document`. Downgrade снимает индекс, но не восстанавливает статусы дублей.

## Redis

`db/redis/client.py` — история чатов и сессии. Учти: история не только
сохраняется, но и **читается обратно при сборке промпта**, поэтому без Redis
пайплайн ответа не деградирует, а падает целиком.
