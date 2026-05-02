from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from db.postgres.models import AdminRole


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8)
    invite_code: Optional[str] = None


class InviteCodeRequest(BaseModel):
    role: AdminRole = AdminRole.admin
    expires_in_hours: Optional[int] = Field(None, ge=1, le=720)


class InviteCodeResponse(BaseModel):
    code: str
    role: AdminRole
    expires_at: Optional[datetime]


class AdminResponse(BaseModel):
    id: str
    username: str
    role: AdminRole
    is_active: bool
    created_at: datetime
    created_by_id: Optional[str]

    model_config = {"from_attributes": True}
