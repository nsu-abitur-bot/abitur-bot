"""Public, whitelisted brand and safe immutable media; no provider settings."""

import re

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.services.branding import favicon_url, media_root
from config.instance import SPECS
from db.postgres.db import get_async_session
from db.postgres.services.instance_settings import InstanceSettingsService

router = APIRouter(prefix="/branding", tags=["branding"])


class BrandingResponse(BaseModel):
    revision: int
    short_name: str
    full_name: str
    logo_url: str
    favicon_url: str
    accent_color: str


@router.get("", response_model=BrandingResponse)
async def get_branding(
    response: Response, session: AsyncSession = Depends(get_async_session)
) -> dict:
    response.headers["Cache-Control"] = "no-store"
    snapshot = await InstanceSettingsService(session).read()
    values = snapshot.values
    nsu = values.get("UNIVERSITY_PROFILE", "nsu") == "nsu"
    logo = values.get("UNIVERSITY_LOGO_URL", "")
    return {
        "revision": snapshot.revision,
        "short_name": values.get("UNIVERSITY_SHORT_NAME")
        or ("НГУ" if nsu else "Ваш вуз"),
        "full_name": values.get("UNIVERSITY_FULL_NAME")
        or ("Новосибирский государственный университет" if nsu else "Ваш вуз"),
        "logo_url": logo,
        "favicon_url": favicon_url(logo),
        "accent_color": values.get("UNIVERSITY_ACCENT_COLOR")
        or SPECS["UNIVERSITY_ACCENT_COLOR"].default,
    }


@router.get("/media/{filename}")
async def get_media(filename: str) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{32}(?:-icon)?\.png", filename):
        raise HTTPException(404)
    path = media_root() / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(
        path,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; sandbox",
        },
    )
