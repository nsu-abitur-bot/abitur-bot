# API для логов сообщений

## Обзор

API предоставляет доступ к детальным логам обработки сообщений бота, включая RAG ретривал и LLM ответы.

## Эндпоинты

### Базовые URL
```
GET /api/v1/logs/
```

### Получение всех логов с фильтрацией

```http
GET /api/v1/logs/?user_id=123&session_id=abc123&message_type=rag_context&limit=50&offset=0
```

**Параметры:**
- `user_id` (optional): ID пользователя для фильтрации
- `session_id` (optional): ID сессии для фильтрации  
- `message_type` (optional): Тип сообщения для фильтрации
- `limit` (default=50, max=1000): Количество записей
- `offset` (default=0): Сдвиг для пагинации

### Получение логов по сессии

```http
GET /api/v1/logs/session/{session_id}?limit=50&offset=0
```

Возвращает все логи для конкретной сессии пользователя.

### Получение логов по пользователю

```http
GET /api/v1/logs/user/{user_id}?limit=50&offset=0
```

Возвращает все логи для конкретного пользователя.

### Получение логов по типу сообщения

```http
GET /api/v1/logs/type/{message_type}?limit=50&offset=0
```

**Типы сообщений:**
- `user_input` - входящие сообщения от пользователей
- `rag_context` - контекст полученный из RAG
- `llm_response` - ответы от LLM
- `faq_match` - совпадения из FAQ

## Структура ответа

```json
{
  "logs": [
    {
      "id": 12345,
      "user_id": 123456789,
      "session_id": "123456789",
      "message_type": "rag_context",
      "content": "НГУ (Новосибирский государственный университет)...",
      "metadata": {
        "sources": ["https://nsu.ru/admission"],
        "context_length": 1250,
        "sources_count": 3
      },
      "created_at": "2026-04-01T10:20:00"
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

## Поля

### MessageLog
- `id`: Уникальный ID записи лога
- `user_id`: ID пользователя Telegram
- `session_id`: ID сессии переписки
- `message_type`: Тип сообщения
- `content`: Содержимое (обрезанное для экономии места)
- `metadata`: Дополнительные данные (источники, длина ответа и т.д.)
- `created_at`: Время создания записи

### Metadata по типам

#### RAG Context
```json
{
  "sources": ["https://nsu.ru/admission", "https://nsu.ru/rules"],
  "context_length": 1250,
  "sources_count": 2
}
```

#### LLM Response
```json
{
  "response_length": 847,
  "provider": "GigaChatLLM"
}
```

#### FAQ Match
```json
{
  "source": "faq"
}
```

#### User Input
```json
{
  "source": "telegram"
}
```

## Примеры использования

### Получить все логи для сессии
```bash
curl "http://localhost:8000/api/v1/logs/session/123456789?limit=100"
```

### Получить только RAG логи
```bash
curl "http://localhost:8000/api/v1/logs/type/rag_context?limit=50"
```

### Получить логи конкретного пользователя
```bash
curl "http://localhost:8000/api/v1/logs/user/123456789?limit=20"
```

## Анализ данных

С помощью этих логов можно анализировать:
1. **Качество RAG** - релевантность найденного контекста
2. **Производительность LLM** - время и длина ответов
3. **Эффективность FAQ** - как часто срабатывают заготовленные ответы
4. **Пути пользователей** - как пользователи взаимодействуют с ботом
5. **Проблемные места** - где бот дает неверные ответы

## Ограничения

- Максимальный `limit`: 1000 записей
- Содержимое `content` обрезается до 2000 символов для экономии места
- Логи хранятся indefinitely (настройте ротацию при необходимости)
