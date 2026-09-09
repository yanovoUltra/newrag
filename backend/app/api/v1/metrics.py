"""运行时观测指标（只读）。"""

from fastapi import APIRouter, Request

from app.core.identity import request_principal, require_roles
from app.store.cache import cache_metrics_snapshot

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/cache")
def cache_metrics(request: Request) -> dict:
    require_roles(request_principal(request), "tenant_admin")
    return cache_metrics_snapshot()
