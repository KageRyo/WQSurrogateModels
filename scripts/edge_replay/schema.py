from __future__ import annotations

from pydantic import BaseModel, Field


class Component(BaseModel):
    key: str
    path: str
    sha256: str
    features: list[str]
    target: str = "Score"


class Condition(BaseModel):
    name: str
    mode: str
    missing_set: str = "complete"
    missing_indicators: list[str] = Field(default_factory=list)
    reconstruction: dict[str, str] = Field(default_factory=dict)


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
    job_id: str
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
