"""Sanitized model and RAG evaluation overview."""

from fastapi import APIRouter, Query, Request

from app.core.identity import request_principal, require_roles, resolve_org
from app.evaluation.service import evaluation_overview

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.get("/overview")
def overview(
    request: Request,
    org_id: str = Query(default="default", min_length=1, max_length=64),
    limit: int = Query(default=12, ge=1, le=30),
) -> dict:
    principal = request_principal(request)
    require_roles(principal, "tenant_admin", "analyst", "viewer")
    return evaluation_overview(resolve_org(principal, org_id), limit=limit)
