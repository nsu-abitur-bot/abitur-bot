# Внешний мониторинг стендов

`tools/monitor.py` запускается раз в минуту **на отдельном постоянно включённом
Linux-хосте**. На сервере приложений только read-only probe. Монитор на том же
хосте не обнаружит отключение питания/сети самого хоста.

Проверки: SSH/Docker, отсутствие/остановка контейнера, unhealthy/starting,
HTTP 200 внешнего API и UI. ERROR/CRITICAL считаются по Docker-логам: последние
5 минут против предыдущих 5. Порог: минимум 5 и рост вдвое, либо ≥20 независимо
от роста. Это число записей ошибок, не процент неуспешных ответов. Дубли логгеров
могут увеличивать число; traceback без заголовка не считается. Нужны сохранённые
Docker-логи минимум за 10 минут. Healthcheck обязателен для обнаружения зависшего
процесса; без него проверяется только running, HTTP отдельно.

Два последовательных неудачных запуска → ALERT; повтор через 30 минут, успешная
проверка → RESOLVED. При ошибке доставки состояние отправки не сохраняется,
следующая проверка повторяет отправку. Состояние сохраняется атомарно между
запусками; systemd не запускает один service параллельно. Сбой самого монитора
или Telegram нельзя сообщить через тот же неисправный канал: контролировать
его timer/service независимым uptime-сервисом хостинга.

## Установка

1. На сервере приложений установить `tools/monitor_probe.py` в
   `/opt/abitur-monitor/monitor_probe.py`, root-owned, не writable для monitor.
   Для каждого стенда создать статический root-owned wrapper, например:

   ```sh
   #!/bin/sh
   exec /usr/local/bin/uv run --no-project --python 3.12 /opt/abitur-monitor/monitor_probe.py /usr/local/bin/abitur-nsu-compose api web bot redis postgres
   ```

   Для исходного Compose добавить scheduler и backup; bot указывать только там,
   где он включён. Сверить все стенды с `/etc/abitur-instances/deploy-targets.json`
   и списком `/usr/local/bin/abitur-*-compose`. Одноразовые permissions/migrate
   не контролировать. Для зависаний добавить healthcheck API `/health` и бота
   `http://127.0.0.1:8080/healthz` в активный Compose overlay.

2. Дать отдельному SSH-пользователю только запуск этого статического wrapper
   через точное правило sudoers, без Docker group и без произвольных аргументов.
   Ключ в authorized_keys ограничить
   `restrict,command="sudo -n /usr/local/sbin/abitur-nsu-probe"`.
   Для нескольких стендов — отдельный ключ/SSH alias с соответствующим forced
   command. Не использовать CI deploy key: он разрешает только деплой.

3. На независимом хосте установить uv и Python 3.12 заранее, создать пользователя
   `abitur-monitor` с домашним каталогом для SSH/uv-cache. Положить `monitor.py`
   в `/opt/abitur-monitor/`, config из примера и token в `/etc/abitur-monitor/`.
   Файлы секрета/config: root:abitur-monitor 0640, каталог 0750. Закрепить host
   key в known_hosts; проверку SSH/TLS не отключать. В SSH config указать IdentityFile
   для каждого alias. Проверить uv path и доступность Python от имени пользователя
   до включения service; ExecStart должен совпадать с реальным путём uv.

4. Создать отдельного Telegram-бота через BotFather; каждый получатель нажимает
   Start. Узнать числовые chat_id через getUpdates этого бота, сохранить в config.
   Username в личных сообщениях Bot API не заменяет chat_id. Токен не хранить в
   git, логах, командной строке или документации.

5. Настроить список целей, публичные HTTP-адреса и ссылку на
   [регламент](../../docs/incident-runbook.md). Скопировать service/timer в
   `/etc/systemd/system/`, выполнить `systemctl daemon-reload` и
   `systemctl enable --now abitur-monitor.timer`. Проверить
   `systemctl status abitur-monitor.timer` и `journalctl -u abitur-monitor.service`.
   Тест доставки (от пользователя monitor, до включения timer):

   ```sh
   uv run --no-project --python 3.12 /opt/abitur-monitor/monitor.py /etc/abitur-monitor/config.json --test
   ```

6. Пройти учебную приёмку из регламента. Одного успешного unit-теста недостаточно.
   Доступы к SSH, админке и панели хостинга должны быть у команды заранее.

## Локальный режим без второго сервера

Если второго хоста нет, допустим `abitur-monitor-local.service` (установить как
`abitur-monitor.service` с тем же timer). Он использует системный Python через uv,
а `probe_command` каждой цели заменяет SSH, например:

```json
{"name": "nsu", "probe_command": ["/usr/local/bin/uv", "run", "--no-project", "--python", "/usr/bin/python3", "/opt/abitur-monitor/tools/monitor_probe.py", "/usr/local/bin/abitur-nsu-compose", "api", "web", "bot", "redis", "postgres"], "urls": ["https://api.nsu.abitur.site/health", "https://nsu.abitur.site/login"]}
```

Файлы tools и docs сохраняют структуру репозитория в `/opt/abitur-monitor`.
Config root-owned 0600: содержит команды, которые сервис запускает от root для
доступа к Docker. У локального варианта отдельный writable uv-cache в StateDirectory.
Установить uv из доверенного источника и использовать имеющийся системный Python.
**Полное падение хоста/его сети остаётся непокрытым.** Это ограничение явно
передать команде; локальная установка не проходит пункт внешней проверки приёмки.
Для двух адресатов `chat_ids` содержит два числовых ID. Состояние уведомления
независимо для каждого, чтобы сбой доставки одному не подавлял алерты другому.
