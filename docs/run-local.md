# Установка локальной копии

Установить uv

```sh
# Linux/macOS
brew install uv

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Установить зависимости

```sh
uv sync
```

Настроить переменные окружения

```sh
# Linux/macOS
cp .env.example .env
# Windows
copy .env.example .env
```

Запустить бота

```sh
uv run python main.py
```

Запуск тестов

```sh
uv run pytest -v
```
