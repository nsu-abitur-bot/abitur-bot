from pydantic import BaseModel, ConfigDict


class SettingsChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int
    changes: dict[str, str | None]


class ConnectionCheck(SettingsChange):
    target: str


class InstanceSettingResponse(BaseModel):
    key: str
    label: str
    group: str
    kind: str
    choices: list[str]
    secret: bool
    configured: bool
    value: str | None


class InstanceSettingsResponse(BaseModel):
    revision: int
    encryption_ready: bool
    settings: list[InstanceSettingResponse]


class ConnectionCheckResponse(BaseModel):
    message: str
