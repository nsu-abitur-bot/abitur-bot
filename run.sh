#!/bin/bash

# Скрипт для запуска Telegram бота с виртуальным окружением

cd "$(dirname "$0")"

echo "🚀 Запуск Telegram бота для НГУ..."
echo ""

# Проверка виртуального окружения
if [ ! -d "venv" ]; then
    echo "❌ Виртуальное окружение не найдено!"
    echo "Создайте его командой: python3 -m venv venv"
    exit 1
fi

# Проверка .env файла
if [ ! -f ".env" ]; then
    echo "❌ Файл .env не найден!"
    echo "Скопируйте .env.example в .env и настройте BOT_TOKEN"
    exit 1
fi

# Проверка Ollama
echo "🔍 Проверка подключения к Ollama..."
if ! curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama сервер не запущен!"
    echo "Запустите Ollama в отдельном терминале:"
    echo "  ollama serve"
    echo ""
    read -p "Продолжить запуск бота? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Ollama работает"
fi

# Активация окружения и запуск
echo "🤖 Запуск бота..."
echo ""
source venv/bin/activate
python bot/main.py
