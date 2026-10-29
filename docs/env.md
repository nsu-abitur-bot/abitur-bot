# Переменные окружения и настройки

Открывай, когда добавляешь env-переменную, меняешь дефолт или решаешь,
где должна жить новая настройка — в env или в таблице `settings`.

Полный актуальный список с дефолтами и комментариями — в `.env.example`
(дефолты в нём синхронизированы с кодом). Эта страница — карта: что где
живёт и что чем перекрывается.

## Где чей дефолт

- env-переменные читаются `os.getenv` по месту использования. Значение в
  `.env.example` — то же, что fallback в коде: расхождение считается багом (#326).
- Дефолты LLM-моделей, работающих с несколькими провайдерами, — в
  `llm/profiles.py` (`VISION_MODEL_DEFAULTS`); `.env.example` ссылается на них.
- Дефолты параметров конкурсных списков (faculty/direction/condition/type) —
  в `parser/config/config.yaml` (`rating.defaults`), в коде не дублируются.
- `DATABASE_URL` полностью перекрывает все `DB_*`. Не задавай её, если
  настраиваешь базу через `DB_HOST`/`DB_PORT`/... — правки `DB_*` действовать
  не будут.

## Настройки, управляемые через таблицу `settings` (веб-админка)

Значения в БД меняются через API админки и применяются **без рестарта**.
env-переменная здесь — только дефолт, если ключа в БД ещё нет (и fallback,
когда БД недоступна).

| Ключ в settings | env-дефолт | Где меняется |
|---|---|---|
| `faq_similarity_threshold` | `FAQ_SIMILARITY_THRESHOLD` | `PUT /api/v1/settings/faq` |
| `crag_enabled` | `CRAG_ENABLED` | `PUT /api/v1/settings/crag` |
| `crag_relevance_threshold` | `CRAG_RELEVANCE_THRESHOLD` | там же |
| `crag_min_chunks` | `CRAG_MIN_CHUNKS` | там же |
| `crag_allow_refine` | `CRAG_ALLOW_REFINE` | там же |
| `crag_use_faculty_table` | `CRAG_USE_FACULTY_TABLE` | там же |
| `crag_max_graded_chunks` | `CRAG_MAX_GRADED_CHUNKS` | там же |
| `rate_limit_system_requests_per_day` | — (дефолт 10000 в коде) | `PUT /api/v1/settings/rate-limit` |
| `rate_limit_user_requests_per_day` | — (дефолт 100 в коде) | там же |
| `admission_scores_last_import` | — | служебная: пишется автозаливкой баллов |

`CRAG_GRADING_CONCURRENCY` остаётся **только env**: влияет на параллелизм
запросов к провайдеру, меняется вместе с инфраструктурой, а не из админки.

## Только env (рестарт требуется)

Секреты и ключи API, выбор провайдеров, параметры БД/Redis/CORS, пайплайн
LightRAG, планировщик конкурсных списков, сиды справочников, логирование —
полный список в `.env.example` по секциям:

- **Провайдеры:** `LLM_PROVIDER`, `LIGHTRAG_LLM_PROVIDER`, `PDF_PARSER_PROVIDER`.
- **LightRAG:** `LIGHTRAG_WORKSPACE_BASE`, `LIGHTRAG_LLM_WORKERS`,
  `LIGHTRAG_EMBEDDING_WORKERS`, `LIGHTRAG_CACHE_CLEAR_TIMEOUT`,
  `LIGHTRAG_STORAGE` (json | postgres; `POSTGRES_WORKSPACE` при этом не задавать).
- **Планировщик списков:** `RATING_DEGREE`, `RATING_TYPES`, `RATING_POLL_*`,
  `RATING_REQUEST_DELAY`, `RATING_FAILED_LISTS_THRESHOLD`,
  `RATING_MAX_SESSION_REFRESHES`, `RATING_USE_MOCK` + `MOCK_RATING_URL`.
- **Сиды:** `SEED_FACULTIES_FORCE`, `SEED_ADMISSION_SCORES`,
  `SEED_ADMISSION_SCORES_FORCE`, `ADMISSION_SCORES_URL`,
  `ADMISSION_SCORES_MAX_AGE_HOURS`.

## Известные ловушки

- `REDIS_URL` без значения падает понятной ошибкой только при создании
  клиента (`db/redis/client.py`); пароль — только буквы/цифры, иначе сломает URL.
- `SECRET_KEY` в примере — заглушка; при деплое обязательно заменить
  (проверки на «не сменили секрет» пока нет — отдельная задача).
- `CORS_ALLOW_ORIGINS` по умолчанию `*` — для прода с админкой сузить
  до явных доменов фронта.
