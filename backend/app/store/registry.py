"""SQLite 登记库：文档/任务 CRUD + 串行持久化阶段文件。"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.entities import Base, ChatRecord, Document, EvalResult, Task

logger = get_logger(__name__)

_engine = None
_session_factory = None


def init_db() -> None:
    global _engine, _session_factory
    settings = get_settings()
    db_path = settings.resolved_registry_db
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    Base.metadata.create_all(_engine)
    logger.info("SQLite registry ready: %s", db_path)


def get_session() -> Session:
    if _session_factory is None:
        raise RuntimeError("registry not initialized, call init_db() first")
    return _session_factory()


# ---------- Document ----------

def create_document(
    filename: str,
    file_type: str,
    org_id: str,
    visibility: str,
    size_bytes: int,
    sha256: str = "",
    fiscal_year: int | None = None,
    fiscal_quarter: int | None = None,
) -> Document:
    with get_session() as s:
        doc = Document(
            filename=filename,
            file_type=file_type,
            org_id=org_id,
            visibility=visibility,
            size_bytes=size_bytes,
            sha256=sha256,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
        )
        s.add(doc)
        s.commit()
        s.refresh(doc)
        return doc


def get_document(doc_id: str) -> Document | None:
    with get_session() as s:
        return s.get(Document, doc_id)


def list_documents(org_id: str | None = None, limit: int = 50, offset: int = 0) -> list[Document]:
    with get_session() as s:
        stmt = select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
        if org_id:
            stmt = stmt.where(Document.org_id == org_id)
        return list(s.scalars(stmt).all())


def update_document(doc_id: str, **fields) -> None:
    with get_session() as s:
        doc = s.get(Document, doc_id)
        if doc is None:
            return
        for k, v in fields.items():
            setattr(doc, k, v)
        s.commit()


def delete_document(doc_id: str) -> None:
    with get_session() as s:
        doc = s.get(Document, doc_id)
        if doc:
            s.delete(doc)
            s.commit()


# ---------- Task ----------

def create_task(doc_id: str, type_: str = "ingest") -> Task:
    with get_session() as s:
        t = Task(doc_id=doc_id, type=type_)
        s.add(t)
        s.commit()
        s.refresh(t)
        return t


def get_task(task_id: str) -> Task | None:
    with get_session() as s:
        return s.get(Task, task_id)


def update_task(task_id: str, **fields) -> None:
    with get_session() as s:
        t = s.get(Task, task_id)
        if t is None:
            return
        for k, v in fields.items():
            setattr(t, k, v)
        s.commit()


# ---------- ChatRecord（问答记录） ----------

def create_chat_record(
    question: str,
    answer: str = "",
    session_id: str = "",
    org_id: str = "default",
    visibility: str = "public",
    citations: list | None = None,
    intent: str = "",
    usage: dict | None = None,
) -> ChatRecord:
    with get_session() as s:
        rec = ChatRecord(
            question=question,
            answer=answer,
            session_id=session_id,
            org_id=org_id,
            visibility=visibility,
            citations=json.dumps(citations or [], ensure_ascii=False),
            intent=intent,
            usage=json.dumps(usage or {}, ensure_ascii=False),
        )
        s.add(rec)
        s.commit()
        s.refresh(rec)
        return rec


def list_chat_records(org_id: str | None = None, limit: int = 200, offset: int = 0) -> list[ChatRecord]:
    with get_session() as s:
        stmt = select(ChatRecord).order_by(ChatRecord.created_at.desc()).limit(limit).offset(offset)
        if org_id:
            stmt = stmt.where(ChatRecord.org_id == org_id)
        return list(s.scalars(stmt).all())


def get_chat_record(record_id: str) -> ChatRecord | None:
    with get_session() as s:
        return s.get(ChatRecord, record_id)


# ---------- EvalResult（评测结果） ----------

def save_eval_result(eval_name: str, scope: str, metrics: dict, payload: dict | None = None) -> EvalResult:
    with get_session() as s:
        rec = EvalResult(
            eval_name=eval_name,
            scope=scope,
            metrics=json.dumps(metrics, ensure_ascii=False),
            payload=json.dumps(payload or {}, ensure_ascii=False),
        )
        s.add(rec)
        s.commit()
        s.refresh(rec)
        return rec


def list_eval_results(eval_name: str | None = None, limit: int = 100) -> list[EvalResult]:
    with get_session() as s:
        stmt = select(EvalResult).order_by(EvalResult.created_at.desc()).limit(limit)
        if eval_name:
            stmt = stmt.where(EvalResult.eval_name == eval_name)
        return list(s.scalars(stmt).all())


# ---------- 串行持久化（断点续跑） ----------

def pipeline_dir(doc_id: str) -> Path:
    return get_settings().resolved_pipeline_dir / doc_id


def save_stage(doc_id: str, stage: str, data: dict) -> Path:
    """落盘阶段中间结果，如 01_layout.json / 04_chunks.json。"""
    p = pipeline_dir(doc_id)
    p.mkdir(parents=True, exist_ok=True)
    idx = {"layout": "01_layout", "ocr": "02_ocr", "sections": "03_sections",
           "chunks": "04_chunks", "vectors": "05_vectors"}.get(stage, stage)
    path = p / f"{idx}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    logger.info("[%s] stage saved: %s", doc_id, path.name)
    return path


def load_stage(doc_id: str, stage: str) -> dict | None:
    p = pipeline_dir(doc_id)
    idx = {"layout": "01_layout", "ocr": "02_ocr", "sections": "03_sections",
           "chunks": "04_chunks", "vectors": "05_vectors"}.get(stage, stage)
    path = p / f"{idx}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stage_done(doc_id: str, stage: str) -> bool:
    return load_stage(doc_id, stage) is not None
