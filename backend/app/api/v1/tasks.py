"""任务状态轮询。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import TaskOut
from app.store.registry import get_task

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskOut)
def get_task_status(task_id: str):
    t = get_task(task_id)
    if not t:
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
