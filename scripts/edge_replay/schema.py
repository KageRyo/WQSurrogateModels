from __future__ import annotations

from pathlib import Path
import re

from pydantic import BaseModel, Field, field_validator


def safe_local_path(value: Path | str) -> Path:
    path = Path(value)
    if ".." in path.parts or any(ord(char) < 32 for char in str(path)):
        raise ValueError("Replay paths must not contain traversal or control characters")
    return path.resolve()


def safe_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("Replay identifiers must contain only letters, digits, underscores and hyphens")
    return value


class Component(BaseModel):
    key: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    path: str
    sha256: str
    features: list[str]
    target: str = "Score"

    @field_validator("path")
    @classmethod
    def relative_artifact(cls, value):
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not value or any(ord(char) < 32 for char in value):
            raise ValueError("Component artifacts must stay inside their packet")
        return value


class Condition(BaseModel):
    name: str
    mode: str
    missing_set: str = "complete"
    missing_indicators: list[str] = Field(default_factory=list)
    reconstruction: dict[str, str] = Field(default_factory=dict)

    @field_validator("reconstruction")
    @classmethod
    def safe_dependencies(cls, value):
        for dependency in value.values():
            safe_identifier(dependency)
        return value


class InputFrame(BaseModel):
    name: str
    kind: str
    n_rows: int
    scenario: str | None = None
    severity: str | None = None
    severity_rank: int | None = None
    perturbation_pct: int | None = None
    multipliers: dict[str, float] = Field(default_factory=dict)


class ReplayJob(BaseModel):
    schema_version: int = 1
    job_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    protocol: str
    seed: int
    model_type: str
    source_artifact: str
    source_sha256: str
    source_origin: str = "archived_workstation_artifact"
    inputs_sha256: str
    frames: list[InputFrame]
    components: list[Component]
    conditions: list[Condition]
    repeats: int = 5
    warmup: int = 1
    windows: int = 107
