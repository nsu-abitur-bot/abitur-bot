import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.services.settings import SettingsService


@pytest.mark.asyncio
async def test_update_faq_settings_roundtrip(session: AsyncSession):
    service = SettingsService(session)

    await service.update_faq_settings(similarity_threshold=0.75)

    assert await service.get_value("faq_similarity_threshold") == "0.75"


@pytest.mark.asyncio
async def test_update_faq_settings_overwrites_previous_value(
    session: AsyncSession,
):
    service = SettingsService(session)

    await service.update_faq_settings(similarity_threshold=0.95)
    await service.update_faq_settings(similarity_threshold=0.80)

    assert await service.get_value("faq_similarity_threshold") == "0.8"
