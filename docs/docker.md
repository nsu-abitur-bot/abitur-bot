# Запуск в Docker

## Предварительные требования

- Docker
- Docker Compose

## Основные команды

### Запуск контейнера

Запуск в фоновом режиме:

```bash
docker-compose up -d
```

Запуск с выводом логов в консоль:

```bash
docker-compose up
```

### Остановка контейнера

```bash
docker-compose down
```

### Пересборка контейнера

Если вы внесли изменения в код или зависимости, нужно пересобрать образ:

```bash
docker-compose up -d --build
```

### Просмотр логов

```bash
docker-compose logs -f bot
```

### Работа с базой данных (PostgreSQL)

Подключение к базе данных из контейнера:

```bash
docker-compose exec postgres psql -U postgres -d abitur
```

## Конфигурация

Убедитесь, что у вас создан файл `.env` с необходимыми переменными окружения (см. `run-local.md`).

Для работы с PostgreSQL в `.env` можно добавить (значения по умолчанию в docker-compose.yml):

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=abitur
```

Если вы используете локальную LLM (например, LM Studio) на хост-машине, убедитесь, что в `docker-compose.yml` или `.env` установлена переменная `LM_API_URL`:

```env
LM_API_URL=http://host.docker.internal:1234/v1
```
