# Структура модулей

Открывай, когда не знаешь, в каком модуле искать нужный код.

```
api/              HTTP API (FastAPI)
  routes/         Эндпоинты по сущностям (faq, abbrev, faculty, admission_score, rag, ...)
  schemas/        Pydantic-схемы запросов/ответов
  services/       Бизнес-логика для роутов (в т.ч. popular_questions.py —
                  кластеризация вопросов по смыслу)
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

## Слои и зависимости

Пакеты разложены по слоям. **Импортировать можно только вниз.** Целевой вид (к
нему идём в задаче #310, сейчас часть пар ещё замкнута друг на друга):

```
L1  abbrev  db  llm (провайдеры)      ничего нашего не импортируют
L2  faq  parser  rag                  пользуются провайдерами и БД
L3  pipeline  evals                   собирают ответ из L1-L2
L4  api  bot                          каналы доступа
L5  scheduler                         периодические задачи
```

Проверяется автоматически: `uv run python tools/check_layers.py`, шаг «Проверка
слоёв» в `pr-checks.yml`. Проверка падает, если появилась пара пакетов, которые
импортируют друг друга и которой нет в `tools/layers_baseline.txt`.

Почему это не формальность: цикл обходят импортом внутри функции, а такой импорт
не видят ни ruff, ни pyrefly — ошибка вылезает только в рантайме, на исполнении
конкретной ветки. Сейчас в прод-коде 25 таких импортов.

**Если проверка упала:**

1. Нужную зависимость передавай параметром, а не импортируй (так `embedder` уже
   приходит в `get_popular_questions` извне).
2. Если код тянет слой выше — он сам принадлежит слою выше, переноси его.
3. Импорт внутри функции ради обхода цикла — не решение, проверка его поймает.

Когда пара исчезла, обнови baseline:

```bash
uv run python tools/check_layers.py --write-baseline
```

Обратно её добавить уже нельзя — в этом смысл. Когда baseline опустеет, файл
удаляем и запрет становится безусловным.
