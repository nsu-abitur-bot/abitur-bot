# 🤖 Telegram бот для НГУ на базе LangChain + Ollama

Телеграм бот-помощник для поступления в Новосибирский Государственный Университет, работающий на локальной LLM модели через Ollama.

## 🚀 Быстрый старт

### 1. Установите зависимости

```bash

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Установите и запустите Ollama

```bash
brew install ollama
ollama pull llama3.2:3b
ollama serve
```

### 3. Настройте переменные окружения

```bash
cp .env.example .env
nano .env
```

### 4. Запустите бота

**Вариант 1: Используя скрипт (рекомендуется)**
```bash
./run.sh
```

**Вариант 2: Вручную**
```bash
ollama serve
source venv/bin/activate
cd bot
python main.py
```

---

## 📁 Структура проекта

```
abitur-bot/
├── bot/
│   └── main.py              # Основной файл Telegram бота
├── llm/
│   └── llm_client.py        # LangChain клиент для работы с LLM
├── .env                     # Конфигурация (не в git)
├── .env.example             # Пример конфигурации
├── requirements.txt         # Зависимости Python
├── run.sh                   # Скрипт быстрого запуска
└── README.md               # Эта инструкция
```

---

## ⚙️ Конфигурация (.env)

```bash
# Telegram Bot Token (получите у @BotFather)
BOT_TOKEN=your_bot_token_here

# Настройки Ollama
LLM_BASE_URL=http://127.0.0.1:11434
LLM_MODEL=llama3.2:3b
LLM_TEMPERATURE=0.8
LLM_TIMEOUT=120

# Системный промпт
SYSTEM_PROMPT=ТЫ LLM помощник для поступления в НГУ...
```

---

## 🎯 Возможности

- ✅ **Контекстная память**: Бот помнит историю диалога для каждого пользователя
- ✅ **Поддержка групп**: Работает в личных чатах и группах
- ✅ **LangChain**: Высокоуровневый фреймворк для работы с LLM
- ✅ **Локальная LLM**: Приватность данных, работа без интернета
- ✅ **Ollama**: Простое управление моделями

---

## 🔧 Требования

- Python 3.9+
- Ollama
- 4GB+ RAM (для llama3.2:3b)
- macOS / Linux / Windows

---

## 📝 Команды Ollama

```bash
# Показать все модели
ollama list

# Скачать модель
ollama pull llama3.2:3b

# Удалить модель
ollama rm llama3.2:3b

# Протестировать модель
ollama run llama3.2:3b

# Запустить сервер
ollama serve
```

---

## 🐛 Решение проблем

### "BOT_TOKEN не задан"
Убедитесь, что в `.env` файле указан токен бота.

### "Connection refused на порту 11434"
Запустите Ollama сервер: `ollama serve`

### "Model not found"
Скачайте модель: `ollama pull llama3.2:3b`

### Медленная работа
Используйте более легкую модель: `ollama pull llama3.2:1b`

---

## 📚 Технологии

- **aiogram** - асинхронный фреймворк для Telegram ботов
- **LangChain** - фреймворк для работы с LLM
- **Ollama** - локальный LLM сервер
- **python-dotenv** - управление переменными окружения

---

## 🎓 Дальнейшее развитие

- [ ] RAG для базы знаний о НГУ
- [ ] Tools для поиска информации
- [ ] Streaming ответов
- [ ] Веб-интерфейс
- [ ] Поддержка документов (PDF, DOCX)

---

## 📄 Лицензия

MIT
