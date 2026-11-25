MAX_USER_MESSAGE_LENGTH = 2000

CHAT_DISCLAIMER = (
    "Бот может ошибаться. Проверяйте важную информацию "
    "в официальных источниках или в приёмной комиссии"
)

# maxapi requires fewer than 4000 characters. Use UTF-16 units for both channels.
WELCOME_MESSAGE_MAX_LENGTH = 3999


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2
