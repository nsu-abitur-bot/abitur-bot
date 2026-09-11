# Структура модулей

Открывай, когда не знаешь, в каком модуле искать нужный код.

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

## Три процесса

- `main.py` — бот (Telegram + MAX)
- `api/main.py` — FastAPI для управления и логов
- `scheduler/main.py` — периодические задачи (конкурсные списки, уведомления)

Запускаются независимо, общаются через PostgreSQL и Redis.
