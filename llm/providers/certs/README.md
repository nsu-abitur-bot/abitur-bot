# Публичные сертификаты для GigaChat

Источники из официальной инструкции Сбера:
https://developers.sber.ru/docs/ru/gigachat/certificates

- https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt
- https://gu-st.ru/content/lending/russian_trusted_sub_ca_pem.crt

Получены по HTTPS с проверкой сертификата 3 ноября 2026. Выпускающий сертификат
проверен через `openssl verify -CAfile <root> <sub>`. Эти сертификаты публичные,
приватных ключей здесь нет. Приложение добавляет их только в SSLContext клиента
GigaChat. Не заменяет certifi/системные CA и не отключает проверку TLS.
