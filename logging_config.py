import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def setup_logging():
    """Настраивает централизованное логирование для всего проекта."""
    
    # Создаем директорию для логов
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Получаем уровень логирования из переменных окружения
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Формат логов
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Настройка корневого логгера
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Удаляем существующие хендлеры чтобы избежать дублирования
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 1. Консольный хендлер
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 2. Файловый хендлер для всех логов с ротацией
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "abitur_bot.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 3. Отдельный файл для логов RAG и LLM с ротацией
    rag_llm_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "rag_llm_detailed.log",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10,
        encoding="utf-8"
    )
    rag_llm_handler.setLevel(logging.DEBUG)
    rag_llm_handler.setFormatter(formatter)
    
    # Добавляем хендлер только для RAG и LLM логгеров
    rag_logger = logging.getLogger("rag.graph_memory")
    llm_logger = logging.getLogger("llm.llm_client")
    bot_logger = logging.getLogger("bot.main")
    
    rag_logger.addHandler(rag_llm_handler)
    llm_logger.addHandler(rag_llm_handler)
    bot_logger.addHandler(rag_llm_handler)
    
    # 4. Файловый хендлер для ошибок с ротацией
    error_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "errors.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    # Устанавливаем уровень для конкретных логгеров
    logging.getLogger("bot.main").setLevel(logging.INFO)
    logging.getLogger("llm.llm_client").setLevel(logging.DEBUG)  # Подробные логи для LLM
    logging.getLogger("rag.graph_memory").setLevel(logging.DEBUG)  # Подробные логи для RAG
    
    # Уменьшаем уровень для слишком подробных внешних библиотек
    logging.getLogger("aiogram").setLevel(logging.WARNING)
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("lightrag").setLevel(logging.INFO)
    
    logging.info("Логирование настроено успешно")
    logging.info(f"Уровень логирования: {log_level}")
    logging.info(f"Директория логов: {log_dir.absolute()}")


def get_logger(name: str) -> logging.Logger:
    """Получает настроенный логгер с указанным именем."""
    return logging.getLogger(name)
