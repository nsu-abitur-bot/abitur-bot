# Установка локальной копии

Создать новое виртуальное окружение

```sh
python -m venv abitur-env
```

Активировать его

```sh
# Linux/macOS
source abitur-env/bin/activate
# windows
abitur-env\Scripts\activate
```

Установить зависимости

```sh
pip install -r requirements.txt
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
python main.py
```
