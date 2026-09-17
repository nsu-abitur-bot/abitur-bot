"""Общая настройка тестов — загружается раньше всех остальных conftest.py."""

import os

# api/auth/security.py падает при импорте без SECRET_KEY: на проде это защита
# от подделки токенов. Тестам настоящий ключ не нужен, а без этой строки
# pytest прерывался бы на сборе целиком у любого, в чьём .env ключа нет.
# setdefault: если ключ задан в окружении, тесты берут его.
os.environ.setdefault(
    "SECRET_KEY", "test-secret-key-not-for-production-0123456789abcdef"
)
