# Пайплайн ответа и RAG

Открывай, когда правишь то, как бот ищет и формирует ответ: `pipeline/llm_client.py`,
`rag/`, `faq/`, `abbrev/` или справочник факультетов.

## Порядок слоёв

`pipeline/llm_client.py:ask_local_llm` — расширение аббревиатур → **FAQ-матчер**
(при попадании отвечает сразу, минуя RAG и LLM) → **RAG** (Corrective RAG
поверх LightRAG, либо обычный LightRAG) → **LLM-генерация** (OpenAI / Gemini) с
доступом к function-calling инструментам.

Порядок не случайный: каждый следующий слой дороже предыдущего, поэтому дешёвый
ответ должен находиться раньше.

## LightRAG

`rag/graph_memory.py` хранит и граф сущностей/связей, и векторный индекс чанков —
единым встроенным стором на диске (`data/lightrag/<graph_id>/`,
`graph_id="abitur_kb"`). Отдельного ChromaDB нет. Чанкинг внутри LightRAG
(`chunk_token_size=400`, overlap 50).

## CRAG

`rag/crag.py` — слой между ретривалом и генерацией:

1. распознаёт факультет и уровень в вопросе по справочнику;
2. сентенс-левел вычищает из чанков предложения про направления **чужих** факультетов;
3. LLM-грейдит релевантность;
4. при нехватке — одна переформулировка и доретрив.

Включается `CRAG_ENABLED` (env) плюс настройки из таблицы `settings`
(веб-админка).

## Справочник факультетов

`Faculty` / `Program`, `db/postgres/services/faculty.py` — авторитетный источник
для CRAG. Редактируется через `/api/v1/faculties`.

При старте выполняется **мягкая** доливка из `db/seed/faculties.json`:
добавляется только недостающее (новые факультеты, алиасы, направления, код у
направления без кода), правки из админки не затираются — поэтому новые записи
сида подхватываются на каждом деплое.

Вручную: `uv run python -m db.seed.load_faculties` (мягко) или `--force`
(жёсткая перезапись, то же делает `SEED_FACULTIES_FORCE=1` при старте).

## Проходные баллы

`AdmissionScore`, `db/postgres/services/admission_score.py`, `parser/scores.py`,
`pipeline/tools/admission_scores.py`.

Числовые баллы вынесены в структурное хранилище и отвечаются function-calling
инструментом `get_admission_scores` из SQL, а не из RAG.

Заливаются автоматически при старте (`init_db.seed_admission_scores`, после
справочника — он нужен для матчинга): парсится страница итогов приёма НГУ и
делается идемпотентный upsert. Чтобы рестарты не дёргали сайт, заливка
пропускается, если последняя была свежее `ADMISSION_SCORES_MAX_AGE_HOURS` (24 ч;
отметка — ключ `admission_scores_last_import` в `settings`). Ошибки не роняют
старт.

Env: `SEED_ADMISSION_SCORES=0` — выключить, `SEED_ADMISSION_SCORES_FORCE=1` —
игнорировать свежесть, `ADMISSION_SCORES_URL` — другой источник.

Вручную из админки: `POST /admission-scores/preview` → просмотр →
`POST /admission-scores/import`.
