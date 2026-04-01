"""
Тесты для MessageService — сервис хранения сообщений в PostgreSQL.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.services.message import MessageService
from db.postgres.services.user import UserService


@pytest.mark.asyncio
async def test_create_message(session: AsyncSession):
    """Тест создания сообщения."""
    user_service = UserService(session)
    await user_service.create_user(user_id=100001)

    msg_service = MessageService(session)
    msg = await msg_service.create_message(
        user_id=100001,
        session_id="100001",
        user_text="Привет, как поступить в НГУ?",
        bot_response="Здравствуйте! Для поступления в НГУ вам нужно...",
    )

    assert msg is not None
    assert msg.user_id == 100001
    assert msg.session_id == "100001"
    assert msg.user_text == "Привет, как поступить в НГУ?"
    assert msg.bot_response == "Здравствуйте! Для поступления в НГУ вам нужно..."
    assert msg.created_at is not None


@pytest.mark.asyncio
async def test_get_messages_by_user(session: AsyncSession):
    """Тест получения сообщений по user_id."""
    user_service = UserService(session)
    await user_service.create_user(user_id=100002)

    msg_service = MessageService(session)
    await msg_service.create_message(
        user_id=100002,
        session_id="100002",
        user_text="Вопрос 1",
        bot_response="Ответ 1",
    )
    await msg_service.create_message(
        user_id=100002,
        session_id="100002",
        user_text="Вопрос 2",
        bot_response="Ответ 2",
    )

    messages = await msg_service.get_messages_by_user(user_id=100002)
    assert len(messages) == 2
    assert messages[0].user_text == "Вопрос 1"
    assert messages[1].user_text == "Вопрос 2"


@pytest.mark.asyncio
async def test_get_messages_by_session(session: AsyncSession):
    """Тест получения сообщений по session_id."""
    user_service = UserService(session)
    await user_service.create_user(user_id=100003)

    msg_service = MessageService(session)
    await msg_service.create_message(
        user_id=100003,
        session_id="chat1:100003",
        user_text="Вопрос в группе",
        bot_response="Ответ в группе",
    )
    await msg_service.create_message(
        user_id=100003,
        session_id="100003",
        user_text="Личный вопрос",
        bot_response="Личный ответ",
    )

    group_msgs = await msg_service.get_messages_by_session(session_id="chat1:100003")
    assert len(group_msgs) == 1
    assert group_msgs[0].user_text == "Вопрос в группе"

    private_msgs = await msg_service.get_messages_by_session(session_id="100003")
    assert len(private_msgs) == 1
    assert private_msgs[0].user_text == "Личный вопрос"


@pytest.mark.asyncio
async def test_get_messages_empty(session: AsyncSession):
    """Тест получения сообщений для пользователя без сообщений."""
    msg_service = MessageService(session)
    messages = await msg_service.get_messages_by_user(user_id=999999)
    assert messages == []


@pytest.mark.asyncio
async def test_get_messages_with_pagination(session: AsyncSession):
    """Тест пагинации при получении сообщений."""
    user_service = UserService(session)
    await user_service.create_user(user_id=100004)

    msg_service = MessageService(session)
    for i in range(5):
        await msg_service.create_message(
            user_id=100004,
            session_id="100004",
            user_text=f"Вопрос {i}",
            bot_response=f"Ответ {i}",
        )

    # Первая страница — 2 записи
    page1 = await msg_service.get_messages_by_user(user_id=100004, limit=2, offset=0)
    assert len(page1) == 2
    assert page1[0].user_text == "Вопрос 0"
    assert page1[1].user_text == "Вопрос 1"

    # Вторая страница
    page2 = await msg_service.get_messages_by_user(user_id=100004, limit=2, offset=2)
    assert len(page2) == 2
    assert page2[0].user_text == "Вопрос 2"

    # За пределами данных
    page_empty = await msg_service.get_messages_by_user(
        user_id=100004, limit=2, offset=10
    )
    assert page_empty == []


@pytest.mark.asyncio
async def test_get_all_messages(session: AsyncSession):
    """Тест получения всех сообщений."""
    user_service = UserService(session)
    await user_service.create_user(user_id=100005)
    await user_service.create_user(user_id=100006)

    msg_service = MessageService(session)
    await msg_service.create_message(
        user_id=100005,
        session_id="100005",
        user_text="Вопрос от user 5",
        bot_response="Ответ для user 5",
    )
    await msg_service.create_message(
        user_id=100006,
        session_id="100006",
        user_text="Вопрос от user 6",
        bot_response="Ответ для user 6",
    )

    all_msgs = await msg_service.get_all_messages()
    assert len(all_msgs) == 2
