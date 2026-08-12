"""运行时观测指标（只读）。"""

from fastapi import APIRouter

from app.store.cache import cache_metrics_snapshot

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/cache")
def cache_metrics() -> dict:
    return cache_metrics_snapshot()
