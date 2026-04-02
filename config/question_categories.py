"""Конфигурация категорий вопросов для абитуриентов."""

from enum import Enum
from typing import Dict, List


class QuestionCategory(str, Enum):
    """Категории вопросов абитуриентов."""
    
    # Основные категории
    ADMISSION = "admission"  # Поступление, правила, сроки
    PROGRAMS = "programs"  # Образовательные программы, специальности
    EXAMS = "exams"  # Экзамены, ЕГЭ, вступительные испытания
    SCORES = "scores"  # Проходные баллы, конкурсы
    DORMS = "dorms"  # Общежития
    DOCUMENTS = "documents"  # Документы, подача заявлений
    COST = "cost"  # Стоимость обучения, льготы
    CAMPUS = "campus"  # Кампус, инфраструктура
    STUDENT_LIFE = "student_life"  # Студенческая жизнь
    CAREER = "career"  # Карьера после выпуска
    TRANSFER = "transfer"  # Перевод, восстановление
    MILITARY = "military"  # Военная кафедра
    INTERNATIONAL = "international"  # Иностранным абитуриентам
    GENERAL = "general"  # Общие вопросы об НГУ
    OFFTOP = "offtop"  # Не относящиеся к НГУ


# Описания категорий для LLM
CATEGORY_DESCRIPTIONS: Dict[QuestionCategory, str] = {
    QuestionCategory.ADMISSION: (
        "Вопросы о поступлении: правила приема, сроки подачи документов, "
        "порядок зачисления, количество мест, целевое обучение"
    ),
    QuestionCategory.PROGRAMS: (
        "Образовательные программы, специальности, направления подготовки, "
        "факультеты, кафедры, учебные планы"
    ),
    QuestionCategory.EXAMS: (
        "Экзамены, ЕГЭ, вступительные испытания, апелляции, "
        "минимальные баллы, дополнительные испытания"
    ),
    QuestionCategory.SCORES: (
        "Проходные баллы, конкурсы, рейтинги, шансы поступления, "
        "конкурсные списки, количество заявлений"
    ),
    QuestionCategory.DORMS: (
        "Общежития: предоставление, условия проживания, стоимость, "
        "очередность, документы для общежития"
    ),
    QuestionCategory.DOCUMENTS: (
        "Документы для поступления: перечень, порядок подачи, сроки, "
        "оригиналы и копии, заявление о согласии"
    ),
    QuestionCategory.COST: (
        "Стоимость обучения, платное и бесплатное образование, льготы, "
        "стипендии, оплата"
    ),
    QuestionCategory.CAMPUS: (
        "Кампус, здания, аудитории, библиотеки, лаборатории, "
        "инфраструктура университета"
    ),
    QuestionCategory.STUDENT_LIFE: (
        "Студенческая жизнь: кружки, секции, мероприятия, традиции, "
        "самоуправление, студенческие организации"
    ),
    QuestionCategory.CAREER: (
        "Карьера после выпуска: трудоустройство, партнеры университета, "
        "стажировки, востребованность выпускников"
    ),
    QuestionCategory.TRANSFER: (
        "Перевод из другого вуза, восстановление, академический отпуск, "
        "отчисление"
    ),
    QuestionCategory.MILITARY: (
        "Военная кафедра, военная подготовка, служба после вуза, "
        "военный учет"
    ),
    QuestionCategory.INTERNATIONAL: (
        "Вопросы иностранных абитуриентов: визы, легализация, "
        "языковые требования, equivalence"
    ),
    QuestionCategory.GENERAL: (
        "Общие вопросы об НГУ: история, достижения, структура, "
        "контакты, расположение"
    ),
    QuestionCategory.OFFTOP: (
        "Вопросы, не относящиеся к НГУ: личные темы, погода, "
        "новости, развлечения, другие вузы"
    ),
}


# Список всех категорий для валидации
ALL_CATEGORIES: List[str] = [cat.value for cat in QuestionCategory]


# Категории, которые считаются относящимися к НГУ
NSU_RELATED_CATEGORIES: List[str] = [
    cat.value for cat in QuestionCategory if cat != QuestionCategory.OFFTOP
]


# Приоритеты категорий для сортировки
CATEGORY_PRIORITY: Dict[QuestionCategory, int] = {
    QuestionCategory.ADMISSION: 1,
    QuestionCategory.PROGRAMS: 2,
    QuestionCategory.EXAMS: 3,
    QuestionCategory.SCORES: 4,
    QuestionCategory.DOCUMENTS: 5,
    QuestionCategory.COST: 6,
    QuestionCategory.DORMS: 7,
    QuestionCategory.CAMPUS: 8,
    QuestionCategory.STUDENT_LIFE: 9,
    QuestionCategory.CAREER: 10,
    QuestionCategory.TRANSFER: 11,
    QuestionCategory.MILITARY: 12,
    QuestionCategory.INTERNATIONAL: 13,
    QuestionCategory.GENERAL: 14,
    QuestionCategory.OFFTOP: 15,
}


def get_category_prompt() -> str:
    """Генерирует промпт для классификации вопросов."""
    categories_text = "\n".join(
        f"{cat.value}: {desc}"
        for cat, desc in CATEGORY_DESCRIPTIONS.items()
    )
    
    return f"""Ты - классификатор вопросов абитуриентов. Определи категорию вопроса.

Категории:
{categories_text}

Правила:
1. Если вопрос касается НГУ - выбери наиболее подходящую категорию из списка выше
2. Если вопрос НЕ касается НГУ (приветствие, благодарность, погода, личные темы) - выбери категорию "offtop"
3. Ответь ТОЛЬКО кодом категории (например: "admission", "programs", "offtop")
4. Не пиши ничего кроме кода категории

Примеры:
- "Какие баллы нужны на программиста?" -> "scores"
- "Привет, как дела?" -> "offtop"
- "Где находится общежитие?" -> "dorms"
- "Расскажи про факультет психологии" -> "programs"
"""


def is_nsu_related(category: str) -> bool:
    """Проверяет, относится ли категория к НГУ."""
    return category in NSU_RELATED_CATEGORIES


def get_category_description(category: str) -> str:
    """Получает описание категории."""
    try:
        cat_enum = QuestionCategory(category)
        return CATEGORY_DESCRIPTIONS[cat_enum]
    except ValueError:
        return "Неизвестная категория"
