# Провайдеры LLM и function-calling инструменты

Открывай, когда добавляешь провайдера, инструмент или правишь параметры генерации.

## Провайдеры

Активные: **OpenAI**, **Gemini**. Выбор через env-переменные:

| Переменная | Назначение | Значения |
|-----------|-----------|---------|
| `LLM_PROVIDER` | Ответы бота | `openai`, `gemini` |
| `LIGHTRAG_LLM_PROVIDER` | LightRAG граф (независимо от бота) | `openai`, `gemini` |
| `PDF_PARSER_PROVIDER` | Парсинг PDF (vision) | `openai`, `gemini` |

Новый провайдер: `providers/<name>.py` плюс
`providers/<name>_graph_adapters.py`, зарегистрировать в `factory.py` и
`graph_memory.py`.

`factory.py: get_llm_provider()` — синглтон, выбирается по `LLM_PROVIDER`.

## Профили параметров

`llm/profiles.py` — по профилю на задачу: `CHAT`, `GRAPH`, `PARSER`, `VISION`,
`INTENT`, `TITLE`, `EMBEDDING`. Там же живут лимиты токенов, поэтому менять
параметры генерации нужно в профиле, а не по месту вызова.

## Function-calling инструменты

- `ToolSpec` (name, description, JSON-schema параметров) и
  `BaseLLMProvider.generate_with_tools(...)` — в `llm/base.py`. Провайдеры
  реализуют нативный tool-loop со стримингом финального ответа.
- Инструменты и диспетчер — в `llm/tools/` (`registry.py`,
  `default_tool_executor`). Исполнитель открывает свою `AsyncSessionLocal`.
- Новый инструмент: добавить `ToolSpec` и `async execute_*` в `llm/tools/`,
  зарегистрировать в `registry.py`, передать в `generate_with_tools` в
  `llm_client.py`.

Принцип: то, что можно посчитать или достать из SQL, должно отвечаться
инструментом, а не генерироваться моделью. Проходные баллы — пример.
