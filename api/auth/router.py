import secrets
from datetime import UTC, datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.db import get_async_session
from db.postgres.models import Admin, AdminRole
from db.postgres.services.admin import AdminService

from .dependencies import get_current_admin, require_superadmin
from .schemas import (
    AdminResponse,
    ChangeRoleRequest,
    InviteCodeRequest,
    InviteCodeResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)
from .security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    session: AsyncSession = Depends(get_async_session),
) -> TokenResponse:
    service = AdminService(session)
    admin = await service.get_by_username(body.username)
    if (
        admin is None
        or not admin.is_active
        or not verify_password(body.password, admin.password_hash)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    return TokenResponse(access_token=create_access_token(admin.id))


@router.post("/register", response_model=AdminResponse, status_code=201)
async def register(
    body: RegisterRequest,
    session: AsyncSession = Depends(get_async_session),
) -> AdminResponse:
    service = AdminService(session)

    total = await service.count()
    is_first = total == 0

    invite = None
    role = AdminRole.superadmin if is_first else AdminRole.admin

    if not is_first:
        if not body.invite_code:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invite code required",
            )
        invite = await service.get_invite_code(body.invite_code)
        if invite is None or invite.used_at is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or already used invite code",
            )
        if invite.expires_at and invite.expires_at < datetime.now(UTC).replace(
            tzinfo=None
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invite code has expired",
            )
        role = AdminRole(invite.role)

    if await service.get_by_username(body.username) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken",
        )

    created_by_id = invite.created_by_id if invite else None
    admin = await service.create(
        username=body.username,
        password_hash=hash_password(body.password),
        role=role,
        created_by_id=created_by_id,
    )

    if invite is not None and body.invite_code:
        await service.use_invite_code(body.invite_code, admin.id)

    await session.commit()
    return AdminResponse.model_validate(admin)


@router.get("/me", response_model=AdminResponse)
async def get_me(current_admin: Admin = Depends(get_current_admin)) -> AdminResponse:
    return AdminResponse.model_validate(current_admin)


@router.post("/invite", response_model=InviteCodeResponse, status_code=201)
async def create_invite(
    body: InviteCodeRequest,
    current_admin: Admin = Depends(require_superadmin),
    session: AsyncSession = Depends(get_async_session),
) -> InviteCodeResponse:
    expires_at = None
    if body.expires_in_hours:
        expires_at = (datetime.now(UTC) + timedelta(hours=body.expires_in_hours)).replace(
            tzinfo=None
        )

    code = secrets.token_urlsafe(16)
    service = AdminService(session)
    invite = await service.create_invite_code(
        code=code,
        created_by_id=current_admin.id,
        role=body.role,
        expires_at=expires_at,
    )
    await session.commit()
    return InviteCodeResponse(
        code=invite.code,
        role=AdminRole(invite.role),
        expires_at=invite.expires_at,
    )


@router.get("/admins", response_model=List[AdminResponse])
async def list_admins(
    _: Admin = Depends(require_superadmin),
    session: AsyncSession = Depends(get_async_session),
) -> List[AdminResponse]:
    admins = await AdminService(session).list_all()
    return [AdminResponse.model_validate(a) for a in admins]


@router.patch("/admins/{admin_id}/deactivate", response_model=AdminResponse)
async def deactivate_admin(
    admin_id: str,
    current_admin: Admin = Depends(require_superadmin),
    session: AsyncSession = Depends(get_async_session),
) -> AdminResponse:
    if admin_id == current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate yourself",
        )
    service = AdminService(session)
    admin = await service.set_active(admin_id, is_active=False)
    if admin is None:
        raise HTTPException(status_code=404, detail="Admin not found")
    await session.commit()
    return AdminResponse.model_validate(admin)


@router.patch("/admins/{admin_id}/role", response_model=AdminResponse)
async def change_admin_role(
    admin_id: str,
    body: ChangeRoleRequest,
    current_admin: Admin = Depends(require_superadmin),
    session: AsyncSession = Depends(get_async_session),
) -> AdminResponse:
    if admin_id == current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own role",
        )

    service = AdminService(session)
    target = await service.get_by_id(admin_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Admin not found")

    # Не даём разжаловать последнего суперадмина, иначе систему некому будет
    # администрировать.
    demoting_superadmin = (
        AdminRole(target.role) == AdminRole.superadmin
        and body.role != AdminRole.superadmin
    )
    if demoting_superadmin and await service.count_superadmins() <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot demote the last superadmin",
        )

    admin = await service.update_role(admin_id, body.role)
    if admin is None:
        raise HTTPException(status_code=404, detail="Admin not found")
    await session.commit()
    return AdminResponse.model_validate(admin)
