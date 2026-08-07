"""阶段一检索：单路稠密检索 + 权限校验。阶段二在此扩展为混合检索。"""

from __future__ import annotations

from app.core.config import get_settings
from app.store import qdrant as qdrant_store


def retrieve(
    query_vector: list[float],
    org_id: str,
    user_visibility: str,
    top_k: int | None = None,
) -> list[dict]:
    """执行检索并做权限二次校验（doc_id + visibility 双重确认）。"""
    settings = get_settings()
    k = top_k or settings.retrieval_top_k
    results = qdrant_store.search_dense(
        query_vector=query_vector,
        org_id=org_id,
        user_visibility=user_visibility,
        top_k=k,
    )
    # 二次校验：仅保留本 org 且可见的条目
    allowed = qdrant_store.visible_levels(user_visibility)
    return [
        r
        for r in results
        if r.get("org_id") == org_id and r.get("visibility") in allowed
    ]
