"""Pydantic DTO。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    filename: str
    file_type: str
    status: str
    org_id: str
    visibility: str
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    chunk_count: int = 0
    size_bytes: int = 0
    error: str | None = None
    created_at: str
    updated_at: str


class TaskOut(BaseModel):
    id: str
    type: str
    doc_id: str
    status: str
    stage: str
    progress: int
    message: str
    created_at: str
    updated_at: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = ""
    org_id: str = Field(min_length=1)
    user_visibility: str = "public"
