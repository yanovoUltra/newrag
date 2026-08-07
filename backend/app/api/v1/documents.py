"""文档上传 / 列表 / 详情 / 删除。"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app.core.config import get_settings
from app.pipelines.ingest import run_ingest
from app.schemas import DocumentOut
from app.store import qdrant as qdrant_store
from app.store.registry import (
    create_document,
    create_task,
    delete_document,
    get_document,
    list_documents,
)
from app.parsers.base import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/documents", tags=["documents"])

VALID_VISIBILITY = {"public", "internal", "restricted"}


@router.post("", response_model=dict, status_code=201)
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    org_id: str = Form(...),
    visibility: str = Form("public"),
    fiscal_year: int | None = Form(None),
    fiscal_quarter: int | None = Form(None),
):
    settings = get_settings()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件类型: {ext or '(无扩展名)'}，支持 {sorted(SUPPORTED_EXTENSIONS)}")
    if visibility not in VALID_VISIBILITY:
        raise HTTPException(400, f"visibility 非法: {visibility}，可选 {sorted(VALID_VISIBILITY)}")

    content = await file.read()
    if not content:
        raise HTTPException(400, "上传内容为空")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"文件超过大小限制 {settings.max_upload_mb}MB")

    upload_dir = settings.resolved_upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = upload_dir / stored_name
    stored_path.write_bytes(content)

    sha256 = hashlib.sha256(content).hexdigest()
    doc = create_document(
        filename=file.filename or stored_name,
        file_type=ext.lstrip("."),
        org_id=org_id,
        visibility=visibility,
        size_bytes=len(content),
        sha256=sha256,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
    )
    task = create_task(doc.id, type_="ingest")
    background.add_task(
        run_ingest,
        doc.id,
        stored_path,
        org_id,
        visibility,
        fiscal_year,
        fiscal_quarter,
        task.id,
    )
    return {"task_id": task.id, "doc_id": doc.id, "status": "pending"}


@router.get("", response_model=list[DocumentOut])
def list_docs(org_id: str | None = None, limit: int = 50, offset: int = 0):
    return [doc_dict(d) for d in list_documents(org_id, limit, offset)]


@router.get("/{doc_id}", response_model=DocumentOut)
def get_doc(doc_id: str):
    d = get_document(doc_id)
    if not d:
        raise HTTPException(404, "文档不存在")
    return doc_dict(d)


@router.delete("/{doc_id}", status_code=200)
def delete_doc(doc_id: str):
    d = get_document(doc_id)
    if not d:
        raise HTTPException(404, "文档不存在")
    try:
        qdrant_store.delete_doc(doc_id)
    except Exception as e:
        raise HTTPException(500, f"向量删除失败: {e}")
    delete_document(doc_id)
    # 清理管线中间产物
    pipeline_target = get_settings().resolved_pipeline_dir / doc_id
    if pipeline_target.exists():
        import shutil

        shutil.rmtree(pipeline_target, ignore_errors=True)
    return {"doc_id": doc_id, "deleted": True}


def doc_dict(d) -> DocumentOut:
    return DocumentOut(
        id=d.id,
        filename=d.filename,
        file_type=d.file_type,
        status=d.status,
        org_id=d.org_id,
        visibility=d.visibility,
        fiscal_year=d.fiscal_year,
        fiscal_quarter=d.fiscal_quarter,
        chunk_count=d.chunk_count,
        error=d.error,
        created_at=d.created_at.isoformat(),
        updated_at=d.updated_at.isoformat(),
    )
