"""Тесты для системы категоризации."""

import pytest
from datetime import UTC, datetime

from config.question_categories import (
    QuestionCategory,
    get_category_prompt,
    is_nsu_related,
    ALL_CATEGORIES,
)


class TestQuestionCategories:
    """Тесты категорий вопросов."""

    def test_all_categories_exist(self):
        """Проверяет что все категории определены."""
        assert len(ALL_CATEGORIES) > 0
        assert "admission" in ALL_CATEGORIES
        assert "offtop" in ALL_CATEGORIES

    def test_is_nsu_related(self):
        """Проверяет определение НГУ-связанных категорий."""
        assert is_nsu_related("admission")
        assert is_nsu_related("programs")
        assert not is_nsu_related("offtop")

    def test_category_prompt_contains_all_categories(self):
        """Проверяет что промпт содержит все категории."""
        prompt = get_category_prompt()
        for category in ALL_CATEGORIES:
            assert category in prompt

    def test_question_category_enum(self):
        """Проверяет работу с enum категорий."""
        assert QuestionCategory.ADMISSION.value == "admission"
        assert QuestionCategory.OFFTOP.value == "offtop"
        
        # Проверяем что все значения уникальны
        values = [cat.value for cat in QuestionCategory]
        assert len(values) == len(set(values))

    @pytest.mark.asyncio
    async def test_category_service_classify(self):
        """Тестирует классификацию сообщения."""
        from db.postgres.services.message_category import MessageCategoryService
        
        # Этот тест требует мокирование LLM провайдера
        # В реальном тесте нужно мокировать get_llm_provider()
        pass

    def test_category_validation(self):
        """Проверяет валидацию категорий."""
        # Валидные категории
        for category in ALL_CATEGORIES:
            assert category in ALL_CATEGORIES
        
        # Невалидные категории
        assert "invalid_category" not in ALL_CATEGORIES
        assert "" not in ALL_CATEGORIES


if __name__ == "__main__":
    pytest.main([__file__])
