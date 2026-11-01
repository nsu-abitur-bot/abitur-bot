# Полное измерение задержки #376

2026-11-01 измерен реальный путь Telegram → BotCore → FAQ/история →
LightRAG/CRAG → генерация/tools → финальная отправка. До: **75,882 с** за
успешный ответ; повтор завершился ошибкой за 12,471 с. После кеширования
словаря tokenizer и переключения **только Chat** на DeepSeek: **31,916 с и
24,736 с**, оба ответа корректны, с источником.

Это совместное изменение кеша и провайдера. Всё ускорение нельзя приписывать
DeepSeek. Только наблюдаемое холодное получение графа сократилось с 42,323 до
**5,411 с**; следующий запрос получил граф за 39,795 мс. Мгновенный ответ и
надёжный SLO не достигнуты и этими четырьмя примерами не доказаны.

## Контур и исходные данные

До: runtime `d407cc8c2942ce406f8b41c11caca9fc0c29a4a7`, Chat/Graph Gemini.
После: runtime `1cb32678cd446e955481e9f95c2039590b001d41`, settings revision5,
сохранена в 12:50:14.265034 UTC: Chat `deepseek-chat`, Graph
`gemini-3.1-flash-lite`; оба embedding-направления — Gemini
`gemini-embedding-001`. CHAT: 1500 токенов, timeout45 с, temperature0.2.
Промпты, правила CRAG, chunk400/overlap50 и embedding-пространство не менялись.

Публичная [страница ФИТ НГУ](https://education.nsu.ru/apply_fit) загружена один
раз штатным authenticated HTTP login → parse → confirm, без ручного
переписывания текста и без вложенных документов. После индексации: 5030 символов,
4 чанка, 44 сущности, 28 связей, LightRAG `processed`. Parse занял 6,963 с,
confirm — 181,281 с: это отдельная индексация, не время ответа.

Владелец дважды до и дважды после отправил вопрос о программах бакалавриата ФИТ.
Он подтвердил оба после-ответа: правильные направление09.03.01 и две программы,
ссылка на источник. История в том же чате сохранялась; повтор не является
идентичным payload. Worker не отправлял сообщений, не запускал второй polling
и не создавал платной/серверной нагрузки в эти окна. Выкладку и смену настроек
выполнил root; Graph/embedding/vision остались на прежних назначениях.

[Исходные JSONL и метаданные](latency-376-data/README.md) содержат только
проверенные числовые spans, случайные request_id и публичную provenance.
Точные UTC-границы, SHA runtime/extractor, хеши файлов и ревизия настроек
сохранены рядом. Пользовательских текстов/Telegram ID/секретов в данных нет.

## Реальные времена этапов

Миллисекунды. Вложенные и параллельные spans **не суммировать**.

| Этап | До, успех | До, ошибка | После, первый | После, повтор |
|---|---:|---:|---:|---:|
| `process_first` | true | false | true | false |
| Telegram total |75882.266|12470.741|31916.454|24736.167|
| Pipeline total |71530.561|10887.256|31250.004|22811.146|
| Typing |2666.412|695.700|243.941|867.990|
| FAQ |656.095|8.549|4098.491|5.431|
| Retrieval total |69146.475|10055.320|25888.158|21343.390|
| Graph acquire |42323.128|43.185|5411.317|39.795|
| Tokenizer внутри acquire |не разделён|не разделён|4289.315|0.306|
| Constructor внутри acquire |не разделён|не разделён|994.830|12.640|
| Storage init внутри acquire |не разделён|не разделён|5.539|0.344|
| Первый поиск, включая keyword LLM/embeddings |9756.433|8544.287|5925.568|4289.138|
| CRAG, включая повторный поиск |17015.261|1449.904|14464.571|16977.167|
| Повторный поиск внутри CRAG |3751.452|нет|9137.671|12106.293|
| Генерация/tools |1668.017|813.077, error|1228.200|1447.036|
| Первый текстовый delta от начала генерации |1549.302|нет|900.779|1139.675|
| Финальная отправка |1241.929|867.899|251.593|1033.025|

Первый model delta отсчитывается **после** retrieval, не от сообщения Telegram.
Доставка первого preview-токена отдельно не измерена. Update age от даты
Telegram: 926/1103/966/159 мс; timestamp имеет секундную точность и включает
очередь/доставку/расхождение часов, это не чистый network RTT.

FAQ в actualDB пуст, FAQ embedding не вызывался. Его await-span включает
планирование event loop и параллельные операции, поэтому первые 4,098 с после
выкладки нельзя назвать стоимостью FAQ-эмбеддинга. Во втором после-запросе есть
constructor/storage spans: RAG-объект реконструировался, но tokenizer уже был
в памяти; это не доказательство попадания в кеш того же graph wrapper.

## Критический путь и вызовы

До первого успеха: 69,146 с retrieval из 75,882 с handler; граф42,323 с —
основная задержка, финальная генерация1,668 с — небольшая часть.

После, повтор: retrieval21,343 с составляет86,3% handler. В нём первый
поиск4,289 с и CRAG16,977 с; внутри CRAG повторный поиск12,106 с, в том числе
Gemini keyword-вызов8,781 с. Два параллельных grading-batch занимают1,353 и
2,269 с; их четыре индивидуальных вызова нельзя складывать как critical path.
Финальная генерация1,447 с не является главным оставшимся bottleneck.

На каждый успешный запрос до и оба после наблюдались одинаковые границы:

| Операция | Количество | Назначение после |
|---|---:|---|
| Intent в фоне |1|Chat/DeepSeek|
| CRAG grade |8, два batch по4|Chat/DeepSeek|
| CRAG refine |1|Chat/DeepSeek|
| Финальная генерация/tool-loop |1|Chat/DeepSeek|
| Graph keyword |2|Graph/Gemini|
| Graph embeddings |6|Gemini|

Итого13 модельных границ +6 embedding spans, из них11 модельных операций
перешли на Chat/DeepSeek. Это **не физическое число HTTP attempts/платных
запросов**. Production SDK retries не инструментированы и остаются `unknown` /
`null`; `tool_execution` в этом cohort не наблюдался. Actual FAQ hit, tool-case
и miss не измерены; offline FAQ/tool-тесты их live latency не заменяют.

В до-ошибке было7 модельных границ и3 embedding spans, три CRAG ClientError
подтверждены как HTTP429; финальная генерация ClientError имеет неизвестный
HTTP-код. CRAG fail-open и отправка fallback объясняют внешние statusok.
В обоих после-запросах ошибок spans нет, кроме ожидаемой отмены одного draft;
владелец подтвердил успешные ответы. Это ограниченная demo-проверка, не оценка
частоты ошибок. Ошибочные12,471 с не усредняются с успешными временами.

## Подтверждённая причина холодного старта и исправление

В старом `rag/graph_memory.py:543` LightRAG создавался синхронно внутри async
получения графа. При пустом дисковом кеше библиотека скачивала словарь tokenizer.
Неплатный локальный cProfile на свежей копии публичной KB показал:

- Graph acquire40,105 с → constructor40,048 с → tokenizer39,842 с →
  `read_file_cached`39,483 с → `requests.get`39,456 с, из них SSL read38,311 с.
- Storage initialize1,755 мс, metadata scan0,126 мс; Python imports2,261 с
  измерены отдельно и не входят в handler/acquire.
- Новый локальный процесс с заполненным кешем и запрещённой загрузкой asset
  из сети: acquire540 мс; повторный in-process acquire2,608 мс.

Это Mac + SSH к тому же серверному proxy, не benchmark сервера с1GB RAM.
Совпадение с реальными42,323 с и перекрывающимися intent timeout42,428 с /
 draft42,320 с подтверждает механизм блокировки event loop. Эти два внешних
await-span не являются чистыми provider/Telegram RTT. Точный подэтап старого
production42,323 с не был инструментирован и остаётся corroborated inference.

Исправление сохраняет **тот же** default tokenizer `gpt-4o-mini`/`o200k_base`
(это счётчик токенов LightRAG, не модель генерации):

1. Docker заранее кеширует проверенный upstream словарь в
   `/app/.cache/tiktoken` от appUID1000. Отдельный interpreter проверяет кеш
   при запрещённом `tiktoken.load.read_file`; hash validation не отключена.
2. Только создание tokenizer вынесено в `asyncio.to_thread`, затем он передан
   `LightRAG(tokenizer=...)`. Graph/storage constructor, initialize/finalize,
   read lease и публикация wrapper остались на owner event loop.
3. Добавлены отдельные tokenizer/constructor/storage spans.

Регрессия responsive-event-loop падает до правки и проходит после; проверены
cancel/release/recovery/contextvars и реальные LightRAG storage/leases.
Root проверил именно конечный image: fresh container UID1000, `--network none`,
read-only + tmpfs, token fetch запрещён, encode проходит за500,385 мс.
Это image-проверка; реальный первый после-acquire всё ещё5,411 с.

Более ранняя правка draft доказана отдельно: fake transport50 мс, n30,
медиана306,84→51,01 мс. Это контролируемая изоляция последовательных RTT,
не прогноз ускорения production и не причина всех прежних30–76 с.

## Следующие шаги и границы результата

Оставшаяся задержка — retrieval/refinement, а не финальная генерация.
Request-scoped reuse Gemini-клиентов можно проверить отдельным controlled
benchmark: сейчас graph adapters создают/закрывают SDK-client на каждый вызов.
Это предложение, а не доказанное ускорение provider compute.

Batch grading, повторное использование результатов grading, изменение min_chunks
или отключение refinement требуют отдельного обсуждения продуктовой логики и
оценки качества (#219). Здесь они не выполнялись; обещания ускорения «в8раз» нет.
Не менялись промпты, правила ответа, embeddings, исходный документ и индексы.
После-тест одновременно меняет кеш и Chat provider, а история растёт; чистое
влияние провайдера на total этими примерами не изолировано. Надёжного p95/SLO нет.

## Корреляция постоянных очередей LightRAG

Независимое ревью нашло P2 в первоначальной телеметрии: priority workers
LightRAG сохраняют ContextVar первого caller. На повторном использовании
того же graph wrapper внутренние `rag_model_call`/`rag_embedding` могли получить
старый request_id. Регрессионный тест создаёт настоящий GraphMemory с файловыми
сторами и очередями, вызывает модель и embedding через vector store из двух
последовательных запросов: до правки оба span относились к первому запросу.

Измерение перенесено с provider callback на внешнее ожидание instance queue.
Теперь эти два span включают **ожидание очереди и выполнение**, а не только
адаптер; callback references хранилищ сохраняют общее wrapping и shutdown.
Настройки workers, provider payload и обработка исключений не меняются.
Исторические JSONL не переписаны: они относятся к прежним runtime и семантике.
В опубликованных окнах все callback IDs совпадают с handler, а graph повторно
конструировался; это не доказывает корректную корреляцию произвольного reused
wrapper. Внешние handler/retrieval/CRAG spans данным дефектом не затронуты.

## Воспроизведение и проверки

`tools.latency_full_profile` запускает полный реальный backend с синтетическим
Telegram transport, свежей собственной PostgreSQL/KB и отдельной Redis, без
polling/отправок. Уникальные DB/output обязательны, старый snapshot отвергается.
SDK ledger сохраняется между процессами; Gemini-only accounting не следует
выдавать за счётчик DeepSeek. Actual cohort выше — отдельные реальные incoming.

`tools.latency_extract` и `tools.latency_error_codes` фильтруют логи на сервере
до передачи по независимому privacy review. Не сохранять raw logs/body/args.
`tools.latency_summarize` не суммирует вложенные этапы и сохраняетunknown counters.

Runtime после: `1cb32678cd446e955481e9f95c2039590b001d41` (reviewed DeepSeek
main4068f98 интегрирован). Локально739pytest passed, ruff/pyrefly/layers passed,
offline regressions=[]; CI41306957095 все jobs green. Независимый Python3.12
review:87focused+9adversarial, реальные offline leases/cancel/reacquire passed.
Финальный docs-only HEAD и его CI фиксируются в PR; он не подменяет SHA runtime.
Полный технический отчёт: `/tmp/abitur-latency-376/full-profile.md`.

Standalone `instrument_sdk` uses the SDK execution context, which may retain the
first caller inside reused LightRAG workers. Its persistent budget ledger and
global attempt count remain valid, but SDK record request_id is not verified
per-request attribution under worker reuse. The queue-boundary fix covers
product rag_model_call/rag_embedding spans only. Historical production cohorts
have no SDK instrumentation and their HTTP attempts/retries remain null.
