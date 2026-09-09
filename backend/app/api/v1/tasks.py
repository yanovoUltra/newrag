"""任务状态轮询。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.core.identity import request_principal, require_roles
from app.schemas import TaskOut
from app.store.registry import get_document, get_task

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskOut)
def get_task_status(task_id: str, request: Request):
    t = get_task(task_id)
    principal = request_principal(request)
    require_roles(principal, "tenant_admin", "analyst", "viewer")
    document = get_document(t.doc_id) if t else None
    if (
        not t
        or not document
        or (principal is not None and document.org_id != principal.org_id)
    ):
        raise HTTPException(404, "任务不存在")
    return TaskOut(
        id=t.id,
        type=t.type,
        doc_id=t.doc_id,
        status=t.status,
        stage=t.stage,
        progress=t.progress,
        message=t.message,
        created_at=t.created_at.isoformat(),
        updated_at=t.updated_at.isoformat(),
    )
