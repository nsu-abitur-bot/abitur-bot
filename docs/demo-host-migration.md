# Перенос демостендов на другой сервер

Для добавления нового вуза без переноса сервера см.
[инструкцию нового демостенда](new-demo-stand.md).

Данные и секреты переносятся отдельно от кода. Новому вузу нужны отдельные Compose
project, БД, Redis, тома, session/encryption keys и администратор. Общими могут быть
проверенные образы и разрешённые владельцем ключи провайдеров; токены ботов разные.

## Подготовка

- Установить Docker Compose и Caddy. Добавление SSH-ключа не требует изменения
  sshd_config; сохранить текущую политику доступа, согласованную с владельцем.
- API/UI публиковать только на loopback; PostgreSQL/Redis не публиковать. Открыть
  SSH/HTTP/HTTPS в firewall, ограничить Docker logs. Не перезаписывать настройки
  системной ротации и автообновлений без необходимости.
- Сохранить infrastructure env, encryption/session keys, image pins, wrappers,
  public CI authorized_keys/forced commands, Caddy storage и конфигурацию. Архивы
  с секретами — только в каталогах 0700 и файлах 0600, не в Git/issue/CI logs.
- Сделать pg_dump -Fc, сохранить LightRAG, документы, branding, логи и Redis RDB.
  Снимок кеша не заменяет PostgreSQL и файловый индекс.
- Использовать проверенные GHCR digests. Если локальный PAT без read:packages,
  Actions GITHUB_TOKEN всё ещё подходит для автодеплоя. Для первичного запуска
  допустим docker save/load работающего образа с проверкой OCI revision под
  локальным migration-тегом; сохранить его до успешной CI-доставки. Не собирать
  приложение на сервере. Большие архивы передавать с докачкой.

## Изолированный запуск

База — `deploy/demo/compose.yaml`, ресурсы — `deploy/demo/runtime.yaml`, последним
идёт общий image-only `images.yaml`. Root-owned wrapper, mode0755:

```sh
#!/bin/sh
exec docker compose --env-file /etc/abitur-instances/nsu.env \
  -f /opt/abitur/backend/deploy/demo/compose.yaml \
  -f /opt/abitur/backend/deploy/demo/runtime.yaml \
  -f /opt/abitur/images.yaml "$@"
```

НГТУ использует другой env и INSTANCE_NAME, например abitur-nstu, API18002/UI13002;
НГУ — API18001/UI13001. Секреты нового проекта генерирует
`uv run python deploy/demo/create_env.py <path> <name> --api-port ... --web-port ...`.
Дополнительные инфраструктурные поля:

```dotenv
API_HOST=api.nstu.example.org
WEB_HOST=nstu.example.org
API_ORIGIN=https://api.nstu.example.org
WEB_ORIGIN=https://nstu.example.org
OUTBOUND_HTTP_PROXY=
```

Прокси по умолчанию выключен. При прямом доступе очистить и сохранённые proxy-поля
приложения, сохранив модели и embedding space. Старому вузу требуется прежний
SETTINGS_ENCRYPTION_KEY для расшифровки настроек, новому — собственный ключ.
Web разрешает свой API_HOST через host-gateway: начальный бренд обращается к
локальному Caddy с правильными Host/TLS, а не старому IP из DNS-кеша. Нужен
действительный сертификат API-vhost; проверку TLS не отключать. Браузер использует
публичный API_ORIGIN.

Восстановить PostgreSQL, файлы — с числовыми UID, выполнить permissions/migrate,
запустить api/web. Новому вузу до публикации задать custom-профиль, имя, логотип и
отключить справочники НГУ. Администратор создаётся локальным bootstrap. Bot service
запускается явно после сохранения собственного токена; кнопка включения канала в
админке не создаёт контейнер. Не переносить чужие документы/FAQ или токен бота.

## Переключение

1. Проверить IP с curl --resolve, TLS, вход и авторизованный маршрут, API и бренд.
   HTTP200 /login недостаточно для проверки авторизации.
2. Остановить старые bot/api для финального dump и файловой копии. Сверить данные,
   запустить новый bot; два polling-процесса одного токена недопустимы.
3. Переключить A-записи web/API обоих вузов. Старый хост временно направляет HTTPS
   на новый IP с правильными Host и tls_server_name в течение TTL. Иначе кеш
   старого API вызывает503 проверки доступа при уже обновлённом frontend. Старую
   БД больше не использовать. Проверить авторитетные DNS и отсутствие старого AAAA.
4. Сохранить новый снимок данных и проверить восстановимость до удаления старого
   хоста. Redis-cache можно начать заново, сохранив RDB для расследования;
   канонические сообщения/документы проверять в PostgreSQL и файловом хранилище.

## Автодеплой и приёмка

Установить helper/forced command из проверенного merged commit по
[CI-инструкции](../deploy/ci/README.md), создать root-owned deploy-targets.json для
работающих проектов. У abitur-deploy только прежние ограниченные sudo-команды
backend/web, без docker group. Public CI-ключи позволяют сохранить прежние private
repository secrets. В обоих репозиториях изменить DEPLOY_HOST и DEPLOY_KNOWN_HOSTS;
host key получить через аутентифицированное SSH, StrictHostKeyChecking не отключать.

Проверить настоящим main workflow обе доставки. Receipt должен соответствовать
main/digest и перечислять экземпляры; проверить каждый API/UI/бот. Устаревший SHA
намеренно отклоняется. Откат приложения не отменяет миграции БД.

Приёмка: разные JWT/учётные данные, отсутствие документов НГУ в НГТУ, отдельные
индексы/media/Redis, корректный бренд и HTTPS, один polling на токен, закрытые
DB/Redis ports, фактическая память и swap. Квоты внешних моделей относятся к
проекту провайдера, перенос IP их не сбрасывает. Смена эмбеддера существующей базы
требует отдельной управляемой переиндексации.
