# Обезличенные исходные замеры #376

Только тайминги, случайные диагностические request_id, статические этапы,
числовые HTTP-коды и метаданные публичного документа. Текстов диалогов,
Telegram/chat/session ID, exception bodies, настроек подключения и ключей нет.

- `telegram-success-120131.jsonl`: реальный Telegram, успешный первый запрос
  после рестарта, 75,882 с, `process_first=true`, один request_id.
- `telegram-failure-120254.jsonl`: реальный повтор с историей, 12,471 с,
  финальная генерация ClientError. Это ошибка, а не успешное ускорение.
- `telegram-cohort-metadata.json`: точные UTC-границы Docker exports, SHA runtime,
  хеш проверенного extractor и хеши исходных файлов.
- `telegram-failure-http-codes.jsonl`: подтверждённые три CRAG HTTP429.
  HTTP-код ошибки финальной генерации не установлен.
- `source-provenance.json`: публичная страница ФИТ НГУ, хеш, время индексации.
  После штатной загрузки: 4 чанка, 44 сущности, 28 связей, processed.
- `local-graph-init.jsonl`: неплатный локальный механизм-эксперимент на Mac,
  пустой собственный tokenizer cache, тот же серверный proxy через SSH.
  Graph acquire 40,105 с, скачивание словаря 39,456 с. Это не Telegram/1GB benchmark.
- `local-graph-warm-cache.jsonl`: новый локальный процесс с заполненным дисковым
  кешем и запрещённой загрузкой token asset из сети: graph acquire 540 мс.
  Повторный in-process acquire 2,608 мс. Model/embedding вызовов нет.

В локальном manifest универсального wrapper упоминается Redis, но эти два
эксперимента были **только graph-init**, Redis не вызывался. В после-правки raw
есть вложенные одноимённые constructor/storage spans старой версии profiler;
их нельзя суммировать. В текущем helper внутренние этапы имеют суффикс `_inner`.

```bash
uv run python -m tools.latency_summarize \
  docs/latency-376-data/telegram-success-120131.jsonl
uv run python -m tools.latency_summarize \
  docs/latency-376-data/telegram-failure-120254.jsonl
```

SDK HTTP attempts/retries в production не наблюдались и выводятся как `null`.
Вложенные/параллельные времена не суммировать. Для каждой исходной категории n=1;
надёжный p95, SLO и частоту ошибок эти данные не оценивают. Модель первого/второго
запроса Gemini, runtime `d407cc8`; словарь и кеш измерены отдельно от генерации.
После: `telegram-after-first-125636.jsonl` — 31,916 с, затем
`telegram-after-second-125711.jsonl` — 24,736 с; оба ответа корректны с источником.
`telegram-after-metadata.json` фиксирует runtime `1cb3267`, revision5,
Chat `deepseek-chat`, Graph/embedding Gemini, границы UTC и хеши.
Это совместный эффект кеша и переключения Chat, не чистый benchmark модели.
Первый реальный graph acquire после исправления — **5,411 с**, не 0,5 с
из локальной/image-проверки; следующий — 39,795 мс.
Данных недостаточно для надёжных p95/SLO/частоты ошибок.
