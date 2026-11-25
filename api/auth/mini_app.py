import hashlib
import hmac
import os
import re
import time
from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import Header, HTTPException


@dataclass(frozen=True)
class MiniAppOwner:
    instance_id: str
    platform: Literal["telegram", "max"]
    external_id: int


def require_mini_app_owner(
    authorization: Annotated[str, Header()] = "",
) -> MiniAppOwner:
    secret = os.getenv("MINI_APP_SHARED_SECRET", "")
    instance_id = os.getenv("MINI_APP_INSTANCE_ID", "")
    if len(secret) < 32 or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", instance_id):
        raise HTTPException(503, "Mini App authentication is not configured")

    match = re.fullmatch(
        r"MiniApp (telegram|max):([1-9][0-9]{0,15}):([0-9]{10}):([0-9a-f]{64})",
        authorization,
    )
    if match is None:
        raise HTTPException(401, "Invalid Mini App session")
    platform, external_id, expires, signature = match.groups()
    claim = f"infovuz:v1:{instance_id}:{platform}:{external_id}:{expires}"
    expected = hmac.new(secret.encode(), claim.encode(), hashlib.sha256).hexdigest()
    now = int(time.time())
    if (
        not now <= int(expires) <= now + 60
        or int(external_id) > 2**53 - 1
        or not hmac.compare_digest(signature, expected)
    ):
        raise HTTPException(401, "Invalid Mini App session")
    return MiniAppOwner(
        instance_id, "telegram" if platform == "telegram" else "max", int(external_id)
    )
