"""Конфигурация администраторов бота."""

import os
from typing import List

from dotenv import load_dotenv

load_dotenv()


def get_admin_ids() -> List[int]:
    """
    Получает список ID администраторов бота.
    
    Returns:
        Список ID администраторов
    """
    admin_ids_str = os.getenv("ADMIN_IDS", "")
    if not admin_ids_str:
        return []
    
    try:
        # Преобразуем строку с ID через запятую в список чисел
        admin_ids = [
            int(id_str.strip()) 
            for id_str in admin_ids_str.split(",") 
            if id_str.strip().isdigit()
        ]
        return admin_ids
    except ValueError:
        return []


def is_admin(user_id: int) -> bool:
    """
    Проверяет, является ли пользователь администратором.
    
    Args:
        user_id: ID пользователя Telegram
        
    Returns:
        True если пользователь является администратором
    """
    return user_id in get_admin_ids()
