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
bootstrap/        Подготовка к старту: схема, миграции, заливка справочников
  init.py         main(): создать БД → миграции → сиды (зовётся из main.py)
  seed.py         Факультеты и проходные баллы
db/
  postgres/       SQLAlchemy ORM + Alembic миграции
    models.py     Все модели (единственный источник правды)
    services/     Сервисы доступа к данным (относительные импорты внутри)
  redis/
    client.py     Клиент Redis (история чатов, сессии)
  seed/           Сид-данные (faculties.json) и их загрузка; когда заливать —
                  решает bootstrap/
evals/            Оценка качества RAG (evaluator + judge)
faq/              Семантический матчер FAQ (matcher.py, эмбеддинги)
llm/              Только провайдеры моделей, без логики ответа
  providers/      openai.py / gemini.py (+ *_graph_adapters.py для LightRAG)
  base.py         BaseLLMProvider, LLMResult, ToolSpec, generate_with_tools
  factory.py      get_llm_provider() — синглтон через LLM_PROVIDER
  profiles.py     Профили параметров (CHAT, GRAPH, PARSER, VISION, INTENT, TITLE, EMBEDDING)
pipeline/         Сборка ответа: слой выше rag/faq/abbrev/db и провайдеров llm
  llm_client.py   ask_local_llm: abbrev → FAQ → RAG → модель(+tools)
  prompts.py      Промпты пайплайна с версиями: правила ответа, хинты RAG, интент
  tools/          Function-calling инструменты (admission_scores.py, registry.py)
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

Пакеты разложены по слоям. **Импортировать можно только вниз.** Кольцевых
импортов между пакетами нет — их разобрали в #310, и проверка не даёт завести
новые:

```
L1  abbrev  db  llm (провайдеры)      ничего нашего не импортируют
L2  faq  parser  rag                  пользуются провайдерами и БД
L3  pipeline  evals                   собирают ответ из L1-L2
L4  api  bot  bootstrap               каналы доступа и подготовка к старту
L5  scheduler                         периодические задачи
```

Проверяется автоматически: `uv run python tools/check_layers.py`, шаг «Проверка
слоёв» в `pr-checks.yml`. Проверка падает, если два пакета импортируют друг
друга.

Почему это не формальность: цикл обходят импортом внутри функции, а такой импорт
не видят ни ruff, ни pyrefly — ошибка вылезает только в рантайме, на исполнении
конкретной ветки. Таких обходов было 25, осталось 12, и все они лежат по другим
причинам: ленивая загрузка SDK провайдера, `TYPE_CHECKING`, локальный stdlib.

**Если проверка упала:**

1. Нужную зависимость передавай параметром, а не импортируй (так `embedder` уже
   приходит в `get_popular_questions` извне).
2. Если код тянет слой выше — он сам принадлежит слою выше, переноси его.
3. Импорт внутри функции ради обхода цикла — не решение, проверка его поймает.

**Осторожно с подменами в тестах.** Пока импорт стоял внутри функции,
`monkeypatch.setattr("llm.factory.get_llm_provider", ...)` работал: имя
разрешалось при каждом вызове. После подъёма на уровень модуля имя связывается
один раз при импорте, и подменять надо его в модуле-потребителе —
`monkeypatch.setattr(crag_module, "get_llm_provider", ...)`. По исходному пути
подмена молча перестаёт действовать, и тест начинает проверять не то.
