"""University identity from the process snapshot backed by instance settings."""

from config.instance import get_setting


def is_nsu() -> bool:
    return get_setting("UNIVERSITY_PROFILE", "nsu") == "nsu"


def identity() -> dict[str, str]:
    defaults = {
        "short_name": "НГУ" if is_nsu() else "Ваш вуз",
        "full_name": "Новосибирский государственный университет"
        if is_nsu()
        else "Ваш вуз",
        "assistant_name": "официальный дружелюбный помощник-бот",
        "website": "https://www.nsu.ru" if is_nsu() else "",
        "email": "",
        "phone": "",
        "address": "",
        "logo_url": "",
    }
    keys = {
        "short_name": "UNIVERSITY_SHORT_NAME",
        "full_name": "UNIVERSITY_FULL_NAME",
        "assistant_name": "ASSISTANT_NAME",
        "website": "UNIVERSITY_WEBSITE",
        "email": "ADMISSIONS_EMAIL",
        "phone": "ADMISSIONS_PHONE",
        "address": "ADMISSIONS_ADDRESS",
        "logo_url": "UNIVERSITY_LOGO_URL",
    }
    return {key: get_setting(keys[key], default) for key, default in defaults.items()}
