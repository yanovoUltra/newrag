"""竞品对标 API 契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BenchmarkRequest(BaseModel):
    org_id: str = Field(min_length=1, max_length=64)
    user_visibility: Literal["public", "internal", "restricted"] = "public"
    companies: list[str] = Field(min_length=2, max_length=5)
    years: list[int] = Field(min_length=1, max_length=5)
    metrics: list[str] = Field(min_length=1, max_length=12)
    name: str = Field(default="", max_length=120)

    @field_validator("companies", "metrics")
    @classmethod
    def unique_text_values(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]
        if len(cleaned) != len(values) or len(set(cleaned)) != len(cleaned):
            raise ValueError("选项不能为空或重复")
        return cleaned

    @field_validator("years")
    @classmethod
    def unique_years(cls, values: list[int]) -> list[int]:
        if len(set(values)) != len(values):
            raise ValueError("财年不能重复")
        return sorted(values)


class BenchmarkMetricOption(BaseModel):
    key: str
    label: str
    kind: str
    display_unit: str
    direction: str
    available_records: int


class BenchmarkCatalog(BaseModel):
    companies: list[str]
    years: list[int]
    metrics: list[BenchmarkMetricOption]


class BenchmarkEvidence(BaseModel):
    doc_id: str
    doc_name: str
    page: int
    section_path: str
    chunk_id: str
    source: str
    raw: str


class BenchmarkObservation(BaseModel):
    company: str
    year: int
    metric: str
    status: Literal["available", "missing"]
    value: float | None = None
    normalized_value: float | None = None
    display_value: str = "—"
    unit: str | None = None
    comparable: bool = False
    conflict: bool = False
    change_value: float | None = None
    change_unit: str | None = None
    evidence: BenchmarkEvidence | None = None
    warnings: list[str] = Field(default_factory=list)


class BenchmarkMetricResult(BaseModel):
    key: str
    label: str
    kind: str
    display_unit: str
    direction: str
    observations: list[BenchmarkObservation]
    insights: list[str]


class BenchmarkStage(BaseModel):
    key: str
    label: str
    status: Literal["complete", "warning"]
    detail: str


class BenchmarkSummary(BaseModel):
    requested_cells: int
    available_cells: int
    comparable_cells: int
    evidenced_cells: int
    coverage: float
    comparable_coverage: float
    evidence_coverage: float
    conflicts: int
    confidence: Literal["high", "medium", "low"]


class BenchmarkAnalysis(BaseModel):
    id: str
    name: str
    generated_at: datetime
    org_id: str
    visibility: str
    companies: list[str]
    years: list[int]
    metrics: list[BenchmarkMetricResult]
    summary: BenchmarkSummary
    insights: list[str]
    warnings: list[str]
    stages: list[BenchmarkStage]


class BenchmarkRunSummary(BaseModel):
    id: str
    name: str
    companies: list[str]
    years: list[int]
    metric_count: int
    coverage: float
    confidence: str
    created_at: datetime
