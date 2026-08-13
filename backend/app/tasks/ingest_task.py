"""ingest 任务（Celery 化）：包装 app.pipelines.ingest.run_ingest。

设计：
- 任务状态继续写 SQLite Task 表（run_ingest 内部 update_task），API /tasks/{id} 轮询不变；
- run_ingest 幂等（stage 落盘可断点续跑），失败由 run_ingest 标记 failed 并抛异常，
  不配置 autoretry（避免业务失败重复入库；瞬时网络故障由嵌入/LLM/rerank 内部重试覆盖）；
- file_path 传字符串（Celery JSON 序列化不支持 Path），任务内转回 Path。
"""

from __future__ import annotations

from pathlib import Path

from app.pipelines.ingest import run_ingest
from app.tasks.celery_app import celery_app


@celery_app.task(name="ingest.run", bind=True)
def ingest_document(
    self,
    doc_id: str,
    file_path: str,
    org_id: str,
    visibility: str,
    fiscal_year: int | None,
    fiscal_quarter: int | None,
    task_id: str | None,
) -> dict:
    """Celery 任务入口：参数与 run_ingest 对齐（file_path 为字符串路径）。"""
    return run_ingest(
        doc_id=doc_id,
        file_path=Path(file_path),
        org_id=org_id,
        visibility=visibility,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        task_id=task_id,
    )
