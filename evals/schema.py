"""Versioned, reviewed input to the product evaluation runner."""

import re
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Assertion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal[
        "contains",
        "not_contains",
        "regex",
        "tool_called",
        "refusal",
        "faq_match",
        "source_contains",
    ]
    value: str | bool

    @model_validator(mode="after")
    def validate_value(self) -> "Assertion":
        boolean = self.kind in {"refusal", "faq_match"}
        if boolean != isinstance(self.value, bool):
            raise ValueError(
                "refusal/faq_match require bool; other assertions require str"
            )
        if isinstance(self.value, str) and not self.value.strip():
            raise ValueError("Empty assertion value")
        if self.kind == "regex":
            try:
                re.compile(str(self.value))
            except re.error as exc:
                raise ValueError("Invalid assertion regex") from exc
        return self


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    source: str = Field(min_length=1)
    approved_by: str | None
    approved_at: date | None
    requires_llm: bool = Field(strict=True)
    assertions: list[Assertion] = Field(default_factory=list)
    judge_criteria: str | None = None
    min_score: int = Field(default=4, ge=1, le=5, strict=True)
    history: list[dict[str, str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_checks(self) -> "Scenario":
        if not self.assertions and not self.judge_criteria:
            raise ValueError("A scenario must have assertions or judge criteria")
        if self.judge_criteria and not self.requires_llm:
            raise ValueError("A judge requires requires_llm=true")
        if bool(self.approved_by) != bool(self.approved_at):
            raise ValueError("Approval needs both reviewer and date")
        return self


def load_scenarios(path: str | Path) -> list[Scenario]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Expected scenario schema_version=1")
    if set(data) != {"schema_version", "scenarios"}:
        raise ValueError("Unexpected scenario document fields")
    raw = data["scenarios"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("Scenario suite must not be empty")
    scenarios = [Scenario.model_validate(item) for item in raw]
    if len({s.id for s in scenarios}) != len(scenarios):
        raise ValueError("Duplicate scenario ids")
    return scenarios
