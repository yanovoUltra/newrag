"""文档上传 / 列表 / 详情 / 删除。"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from datetime import date
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile

from app.core.config import get_settings
from app.pipelines.ingest import run_ingest
from app.schemas import DocumentOut, DocumentPage
from app.store import qdrant as qdrant_store
from app.store.registry import (
    create_document,
    create_task,
    delete_document,
    delete_fields,
    find_document_by_filename,
    find_document_by_sha256,
    get_document,
    query_documents_page,
    update_document,
)
from app.parsers.base import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/documents", tags=["documents"])

VALID_VISIBILITY = {"public", "internal", "restricted"}
MIN_FISCAL_YEAR = 1990


def _current_year(today: date | None = None) -> int:
    """返回服务器当前自然年；允许注入日期以便确定性测试。"""
    return (today or date.today()).year


def _same_file_key(a: str, b: str) -> bool:
    """同名判定：忽略目录与扩展名大小写，比较文件名主干。"""
    return Path(a).stem.lower() == Path(b).stem.lower()


def _purge_document(doc_id: str) -> None:
    """彻底删除文档全部痕迹：向量点 + 字段索引 + registry 记录 + 管线中间产物（用户显式删除用）。"""
    try:
        qdrant_store.delete_doc(doc_id)
    except Exception as e:
        raise HTTPException(500, f"旧版向量删除失败: {e}")
    try:
        delete_fields(doc_id)
    except Exception:
        pass  # 字段索引删除失败不阻断主流程
    delete_document(doc_id)
    pipeline_target = get_settings().resolved_pipeline_dir / doc_id
    if pipeline_target.exists():
        shutil.rmtree(pipeline_target, ignore_errors=True)


def _archive_document(doc_id: str) -> None:
    """版本归档（同名替换用）：删向量点防检索命中旧版，registry 记录标 archived 保留
    （上传文件与管线中间产物不清除，可追溯/可回滚），新版本正常入库。"""
    try:
        qdrant_store.delete_doc(doc_id)
    except Exception as e:
        raise HTTPException(500, f"旧版向量删除失败: {e}")
    try:
        delete_fields(doc_id)
    except Exception:
        pass
    update_document(doc_id, status="archived")


def _find_replace_target(settings, org_id: str, sha256: str, filename: str):
    """陈旧文档防护：
    - 同 org + 同 sha256 → 返回 ("duplicate", doc)：幂等，不重复入库（failed 状态除外，允许重试）；
    - 同 org + 同名文件且内容不同 → 返回 ("replace", doc)：替换旧版，先归档旧点再入库。
    已归档（archived）记录视为不存在：跳过，允许重新入库形成新版本（旧版留档）。
    返回 (kind, doc|None)。"""
    same_content = find_document_by_sha256(org_id, sha256)
    if same_content is not None:
        if same_content.status == "failed":
            return "replace", same_content
        return "duplicate", same_content
    if settings.upload_replace_same_filename:
        same_name = find_document_by_filename(org_id, filename)
        if same_name is not None:
            return "replace", same_name
    return "none", None


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
    if fiscal_year is not None:
        current_year = _current_year()
        if not MIN_FISCAL_YEAR <= fiscal_year <= current_year:
            raise HTTPException(
                400,
                f"fiscal_year 非法: {fiscal_year}，允许范围 {MIN_FISCAL_YEAR}-{current_year}",
            )

    upload_dir = settings.resolved_upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    digest = hashlib.sha256()
    size_bytes = 0
    temp = tempfile.NamedTemporaryFile(prefix="upload-", suffix=".part", dir=upload_dir, delete=False)
    temp_path = Path(temp.name)
    try:
        with temp:
            while chunk := await file.read(1024 * 1024):
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise HTTPException(413, f"文件超过大小限制 {settings.max_upload_mb}MB")
                digest.update(chunk)
                temp.write(chunk)
        if size_bytes == 0:
            raise HTTPException(400, "上传内容为空")
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    sha256 = digest.hexdigest()

    # ---- 陈旧文档防护：sha256 幂等 / 同名替换（旧版归档留档）----
    kind, existing = _find_replace_target(settings, org_id, sha256, file.filename or "")
    if kind == "duplicate" and existing is not None:
        temp_path.unlink(missing_ok=True)
        return {"task_id": "", "doc_id": existing.id, "status": "duplicate"}
    if kind == "replace" and existing is not None:
        _archive_document(existing.id)

    stored_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = upload_dir / stored_name
    os.replace(temp_path, stored_path)

    doc = create_document(
        filename=file.filename or stored_name,
        file_type=ext.lstrip("."),
        org_id=org_id,
        visibility=visibility,
        size_bytes=size_bytes,
        sha256=sha256,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
    )
    task = create_task(doc.id, type_="ingest")
    # 阶段四：任务后端可切换——celery 走独立 worker（broker=Redis），background 保持进程内线程池
    settings = get_settings()
    if settings.task_backend == "celery":
        from app.tasks.ingest_task import ingest_document

        ingest_document.delay(
            doc.id, str(stored_path), org_id, visibility, fiscal_year, fiscal_quarter, task.id
        )
    else:
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
def list_docs(
    org_id: str | None = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    include_archived: bool = False,
):
    docs, _ = query_documents_page(
        org_id=org_id,
        limit=limit,
        offset=offset,
        include_archived=include_archived,
    )
    return [doc_dict(d) for d in docs]


@router.get("/page", response_model=DocumentPage)
def list_docs_page(
    org_id: str | None = None,
    filename: str | None = None,
    status: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    include_archived: bool = False,
):
    docs, total = query_documents_page(
        org_id=org_id,
        filename=filename,
        status=status,
        limit=limit,
        offset=offset,
        include_archived=include_archived,
    )
    return DocumentPage(
        items=[doc_dict(d) for d in docs],
        total=total,
        limit=limit,
        offset=offset,
    )


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
    _purge_document(doc_id)
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
        size_bytes=d.size_bytes,
        error=d.error,
        created_at=d.created_at.isoformat(),
        updated_at=d.updated_at.isoformat(),
    )
