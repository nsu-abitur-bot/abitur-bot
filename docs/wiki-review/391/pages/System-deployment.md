# Развёртывание системы

Состояние инструкций сверено 01.11.2026 по backend `c608c31` и frontend `ac65b37`.
Сервисы на сервере в рамках этой сверки не запускались.

## Конфигурация

Backend получает начальные настройки через окружение. Состав параметров и
значения по умолчанию описаны в
[`config/instance.py`](https://github.com/nsu-abitur-bot/abitur-bot/blob/main/config/instance.py).
Настройки экземпляра, доступные через панель, и начальная настройка описаны в
[документации демостендов](https://github.com/nsu-abitur-bot/abitur-bot/blob/main/docs/demo-instances.md).

| Группа | Параметры и назначение |
|---|---|
| Авторизация | `SECRET_KEY` — секрет подписи JWT; первый superadmin создаётся отдельной CLI-командой |
| Telegram/MAX | `BOT_TOKEN`, `MAX_BOT_TOKEN`, `TELEGRAM_ENABLED`, `MAX_ENABLED`; для Telegram также параметры прокси и polling-lock |
| PostgreSQL | `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`; Compose задаёт адреса сервисов своей сети |
| Redis | `REDIS_URL`; корневой Compose требует `REDIS_PASSWORD` |
| Провайдеры | `LLM_PROVIDER`, `FAQ_EMBEDDING_PROVIDER`, `LIGHTRAG_LLM_PROVIDER`, `LIGHTRAG_EMBEDDING_PROVIDER`, `PDF_PARSER_PROVIDER` |
| Модели | API-ключи и имена моделей OpenAI/Gemini, параметры вызовов и [профили лимитов](Лимиты-LLM) |
| Образ backend | `BACKEND_IMAGE_TAG` — выбранный тег образа GHCR |
| Тесты | `TEST_DB_NAME` и отдельные подключения к тестовым сервисам |
| Журналирование | `LOG_LEVEL`, `LOG_DIR`; ограничить доступ к журналам и резервным копиям |

Реальные значения секретов хранятся в закрытой конфигурации развёртывания.
В документацию, Git и команды с записью в историю они не помещаются.

## Контейнеры backend

В каталоге `abitur-bot` после настройки окружения:

```sh
docker compose pull
docker compose up -d
```

Корневой `docker-compose.yml` использует образ
`ghcr.io/nsu-abitur-bot/abitur-bot` с выбранным `BACKEND_IMAGE_TAG`.
Сервисы: `bot`, `api`, `scheduler`, `postgres`, `redis`.
`bot` и `api` используют общий каталог данных LightRAG. Резервировать нужно
SQL и соответствующие ему файлы графа/индекса.

PostgreSQL/Redis находятся во внутренней сети Compose. API сейчас публикует
`8000:8000` на интерфейсах хоста. Для публичного сервера требуется HTTPS-прокси
и ограничение прямого доступа к backend. Конфигурация конкретного сервера
проверяется отдельно.

Бот при старте инициализирует БД, применяет миграции и может загружать справочники.
Порядок запуска и настройки заполнения данных согласуются для выбранного стенда.

## Первый администратор

После миграций на доверенном хосте с настройками нужной БД:

```sh
uv run python -m api.auth.bootstrap --username admin
```

Пароль вводится скрыто с подтверждением. Команда работает при пустой таблице
администраторов; существующие аккаунты не сбрасывает. Дальнейшая регистрация
через `POST /api/v1/auth/register` требует одноразового приглашения.
Подробности и команды для Docker — в
[инструкции авторизации](https://github.com/nsu-abitur-bot/abitur-bot/blob/main/docs/admin-auth.md)
и документации демостендов.

## Обновление и остановка

Перед обновлением выберите проверенную версию образа, сохраните совместимые
резервные копии и подготовьте ручное восстановление по [плану релиза](План-релиза).
После обновления проверьте вход сотрудников и ответы в обоих ботах.

```sh
docker compose pull
docker compose up -d
```

Остановка:

```sh
docker compose down
```

Workflow `publish-image.yml` собирает и публикует backend в GHCR при push в main.
Полная автоматическая выкладка на сервер и приёмка после неё — отдельная работа
релиза 2. Публикацию образа проверяйте по Actions, запуск конкретного стенда —
по его журналу выкладки и проверкам.

## Панель сотрудников

`abitur-web` разворачивается отдельно по
[README](https://github.com/nsu-abitur-bot/abitur-web/blob/main/README.md) и
[инструкции Docker](https://github.com/nsu-abitur-bot/abitur-web/blob/main/docs/docker.md).
Настройте адрес backend и закрытые параметры сессий/БД панели. Порты и домены
выбранного варианта проверьте по его Compose и настройкам HTTPS.

[На главную](Home) · [Локальный запуск](Installing-a-local-copy) · [SAD](SAD)
