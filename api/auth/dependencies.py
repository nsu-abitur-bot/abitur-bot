from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.db import get_async_session
from db.postgres.models import Admin, AdminRole
from db.postgres.services.admin import AdminService

from .security import decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

_unauthorized = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_admin(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_async_session),
) -> Admin:
    try:
        payload = decode_access_token(token)
        admin_id = payload.get("sub")
        if not admin_id:
            raise _unauthorized
    except InvalidTokenError:
        raise _unauthorized

    admin = await AdminService(session).get_by_id(admin_id)
    if admin is None or not admin.is_active:
        raise _unauthorized
    return admin


async def require_admin(admin: Admin = Depends(get_current_admin)) -> Admin:
    """Требует роль admin или superadmin (не viewer)."""
    if admin.role == AdminRole.viewer:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return admin


async def require_superadmin(admin: Admin = Depends(get_current_admin)) -> Admin:
    """Требует роль superadmin."""
    if admin.role != AdminRole.superadmin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return admin
