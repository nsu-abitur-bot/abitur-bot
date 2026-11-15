# Локальный запуск

Инструкция для разработчика. Команды backend выполняются из `abitur-bot`,
команды панели - из `abitur-web`. Для разработки нужны отдельные PostgreSQL,
Redis и настройки тестовых ботов/провайдеров. Команды ниже предназначены
для локального окружения, их не выполняют на работающем общем стенде.

## Серверная часть без Docker

Установите [uv](https://docs.astral.sh/uv/getting-started/installation/)
и Python >=3.12. Синхронизируйте зависимости:

```sh
uv sync
```

Настройте локальное окружение по [инструкции развёртывания](System-deployment).
Имена и описание настроек хранятся в исходниках. Реальные пароли и токены
передаются локально и не коммитятся.

Каждый компонент запускается в отдельном терминале:

```sh
# Боты; при старте выполняется инициализация БД
uv run python main.py

# API панели, доступный только с этого компьютера
uv run uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# Планировщик конкурсных списков и уведомлений
uv run python -m scheduler.main
```

Старт бота применяет миграции и может заполнять справочники. Перед запуском
проверьте, что выбрана локальная БД. Первого администратора создают отдельной
командой, описанной на странице развёртывания.

## Проверки серверной части

```sh
uv run pytest -v
```

Интеграционные тесты создают и пересоздают таблицы. Используйте отдельный
тестовый PostgreSQL, проверьте `DB_*` и `TEST_DB_NAME`. Имя тестовой базы
должно отличаться от рабочей. До запуска изучите fixtures нужных тестов.

Изолированная проверка сборки ответов описана в
[evals/README.md](https://github.com/nsu-abitur-bot/abitur-bot/blob/main/evals/README.md).
Она выполняется отдельным процессом с `PYTHON_DOTENV_DISABLED=1` и `--offline`.
Живой прогон использует модели и хранилища и требует отдельного тестового окружения.

## Серверная часть через Docker

Корневой Compose использует готовый образ backend из GHCR:

```sh
docker compose pull
docker compose up -d
```

Запускаются `bot`, `api`, `scheduler`, `postgres`, `redis`.
API опубликован на порту 8000. Swagger - `/docs`. PostgreSQL и Redis доступны
внутри сети Compose, их порты на хост не публикуются.

```sh
docker compose down
```

Команда останавливает выбранный локальный Compose-проект и сохраняет тома.
Перед её выполнением проверьте каталог и имя проекта.

## Панель сотрудников

Версии инструментов закреплены в `abitur-web/.mise.toml`: Node.js 24.18.1,
Bun 1.4.4, Task 3.54.0 и Lefthook 2.2.2.

```sh
mise trust
mise install
mise exec -- bun install --frozen-lockfile
mise exec -- task dev
```

Перед `task dev` задайте локальные настройки по
[README панели](https://github.com/nsu-abitur-bot/abitur-web/blob/main/README.md).
`task check` выполняет проверки типов, стиля и модульные тесты. `task build`
готовит production-сборку. Для каждого worktree устанавливается собственный
`node_modules`.

Docker-вариант панели описан в
[docs/docker.md](https://github.com/nsu-abitur-bot/abitur-web/blob/main/docs/docker.md).

[На главную](Home) · [Развёртывание](System-deployment)
