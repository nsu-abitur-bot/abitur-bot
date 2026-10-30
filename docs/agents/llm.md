# Провайдеры LLM и function-calling инструменты

Открывай, когда добавляешь провайдера, инструмент или правишь параметры генерации.

## Провайдеры

Активные: **OpenAI**, **Gemini**. Выбор и ключи — в админке,
«Настройки экземпляра» (superadmin). Ключи параметров:

| Переменная | Назначение | Значения |
|-----------|-----------|---------|
| `LLM_PROVIDER` | Ответы бота | `openai`, `gemini` |
| `LIGHTRAG_LLM_PROVIDER` | LightRAG граф (независимо от бота) | `openai`, `gemini` |
| `FAQ_EMBEDDING_PROVIDER` | Эмбеддинги FAQ | `openai`, `gemini` |
| `LIGHTRAG_EMBEDDING_PROVIDER` | Эмбеддинги графа | `openai`, `gemini` |
| `PDF_PARSER_PROVIDER` | Парсинг PDF (vision) | `openai`, `gemini` |

Новый провайдер: `providers/<name>.py` плюс
`providers/<name>_graph_adapters.py`, зарегистрировать в `factory.py` и
`graph_memory.py`.

`factory.py: get_llm_provider()` кэширует клиент по конфигурации подключения.
`get_faq_provider()` выбирает отдельного провайдера эмбеддингов FAQ.
Потребители читают `config.instance.get_setting`, снимок БД обновляется
в каждом процессе. Порядок перехода с env и хранение секретов —
[настройки экземпляра](../instance-settings.md).

## Профили параметров

`llm/profiles.py` — по профилю на задачу: `CHAT`, `GRAPH`, `PARSER`, `VISION`,
`INTENT`, `TITLE`, `EMBEDDING`. Значения профилей редактируются через админку (`PROFILE_*`),
исходные дефолты остаются здесь. Лимиты токенов нужно менять
параметры генерации нужно в профиле, а не по месту вызова.

## Function-calling инструменты

- `ToolSpec` (name, description, JSON-schema параметров) и
  `BaseLLMProvider.generate_with_tools(...)` — в `llm/base.py`. Провайдеры
  реализуют нативный tool-loop со стримингом финального ответа.
- Инструменты и диспетчер — в `pipeline/tools/` (`registry.py`,
  `default_tool_executor`). Исполнитель открывает свою `AsyncSessionLocal`.
- Новый инструмент: добавить `ToolSpec` и `async execute_*` в `pipeline/tools/`,
  зарегистрировать в `registry.py`, передать в `generate_with_tools` в
  `llm_client.py`.

Принцип: то, что можно посчитать или достать из SQL, должно отвечаться
инструментом, а не генерироваться моделью. Проходные баллы — пример.
