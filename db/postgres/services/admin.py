from datetime import UTC, datetime
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.models import Admin, AdminRole, InviteCode


class AdminService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(Admin))
        return result.scalar_one()

    async def get_by_id(self, admin_id: str) -> Optional[Admin]:
        result = await self.session.execute(select(Admin).where(Admin.id == admin_id))
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> Optional[Admin]:
        result = await self.session.execute(
            select(Admin).where(Admin.username == username)
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        username: str,
        password_hash: str,
        role: AdminRole,
        created_by_id: Optional[str] = None,
    ) -> Admin:
        admin = Admin(
            username=username,
            password_hash=password_hash,
            role=role,
            created_by_id=created_by_id,
        )
        self.session.add(admin)
        await self.session.flush()
        await self.session.refresh(admin)
        return admin

    async def list_all(self) -> list[Admin]:
        result = await self.session.execute(select(Admin).order_by(Admin.created_at))
        return list(result.scalars().all())

    async def set_active(self, admin_id: str, is_active: bool) -> Optional[Admin]:
        admin = await self.get_by_id(admin_id)
        if admin is None:
            return None
        admin.is_active = is_active
        await self.session.flush()
        return admin

    async def update_role(self, admin_id: str, role: AdminRole) -> Optional[Admin]:
        admin = await self.get_by_id(admin_id)
        if admin is None:
            return None
        admin.role = role
        await self.session.flush()
        return admin

    async def count_superadmins(self) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(Admin)
            .where(Admin.role == AdminRole.superadmin.value)
        )
        return result.scalar_one()

    async def create_invite_code(
        self,
        code: str,
        created_by_id: str,
        role: AdminRole,
        expires_at: Optional[datetime] = None,
    ) -> InviteCode:
        invite = InviteCode(
            code=code,
            created_by_id=created_by_id,
            role=role,
            expires_at=expires_at,
        )
        self.session.add(invite)
        await self.session.flush()
        await self.session.refresh(invite)
        return invite

    async def get_invite_code(self, code: str) -> Optional[InviteCode]:
        result = await self.session.execute(
            select(InviteCode).where(InviteCode.code == code)
        )
        return result.scalar_one_or_none()

    async def use_invite_code(self, code: str, used_by_id: str) -> None:
        invite = await self.get_invite_code(code)
        if invite is not None:
            invite.used_by_id = used_by_id
            invite.used_at = datetime.now(UTC).replace(tzinfo=None)
            await self.session.flush()
