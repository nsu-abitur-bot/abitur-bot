import os
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from dotenv import load_dotenv

load_dotenv()

# Фиксированный дефолт позволял бы подделать JWT админа и получить доступ к API,
# поэтому отсутствие/placeholder SECRET_KEY обязаны ломать старт, а не молча работать
_INSECURE_SECRET_KEYS = frozenset(
    {
        "",
        "change-me-in-production-use-a-long-random-string",
        "change-me-to-a-long-random-string",
    }
)

# HS256 подписывает ключом произвольной длины, поэтому короткий ключ перебирается
# офлайн по любому выданному токену. 32 байта — минимум для HMAC-SHA256.
MIN_SECRET_KEY_BYTES = 32

SECRET_KEY = os.getenv("SECRET_KEY", "")
if SECRET_KEY in _INSECURE_SECRET_KEYS:
    raise RuntimeError(
        "SECRET_KEY не задан или равен placeholder из .env.example. "
        'Сгенерируйте: python -c "import secrets; print(secrets.token_hex(32))"'
    )
if len(SECRET_KEY.encode()) < MIN_SECRET_KEY_BYTES:
    raise RuntimeError(
        f"SECRET_KEY короче {MIN_SECRET_KEY_BYTES} байт — такой ключ можно подобрать. "
        'Сгенерируйте: python -c "import secrets; print(secrets.token_hex(32))"'
    )
ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 7


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(admin_id: str) -> str:
    payload = {
        "sub": admin_id,
        "exp": datetime.now(UTC) + timedelta(days=TOKEN_EXPIRE_DAYS),
        "iat": datetime.now(UTC),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
