"""OIDC 身份、租户和角色授权。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jwt
from fastapi import HTTPException, Request

from app.core.config import get_settings

VISIBILITY_LEVELS = {"public": 0, "internal": 1, "restricted": 2}


class IdentityError(ValueError):
    """Bearer token 无效或缺少必要租户声明。"""


@dataclass(frozen=True)
class Principal:
    subject: str
    org_id: str
    roles: frozenset[str]
    visibility: str


@lru_cache(maxsize=8)
def _jwk_client(url: str, lifespan: int, timeout: float) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(
        url,
        cache_keys=True,
        cache_jwk_set=True,
        lifespan=lifespan,
        timeout=timeout,
    )


def authenticate_bearer(token: str) -> Principal:
    settings = get_settings()
    issuer = settings.oidc_issuer.rstrip("/")
    jwks_url = settings.oidc_jwks_url or f"{issuer}/protocol/openid-connect/certs"
    if not issuer or not settings.oidc_audience:
        raise IdentityError("OIDC 服务端配置不完整")
    try:
        signing_key = _jwk_client(
            jwks_url,
            settings.oidc_jwks_cache_seconds,
            settings.oidc_http_timeout,
        ).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=settings.oidc_audience,
            issuer=issuer,
            leeway=30,
            options={"require": ["exp", "iat", "sub"]},
        )
    except Exception as exc:  # PyJWT/JWKS 错误统一对外隐藏
        raise IdentityError("Bearer token 无效") from exc

    subject = str(claims.get("sub") or "").strip()
    org_id = str(_claim_at_path(claims, settings.oidc_org_claim) or "").strip()
    roles_raw = _claim_at_path(claims, settings.oidc_roles_claim)
    visibility = str(
        _claim_at_path(claims, settings.oidc_visibility_claim) or "public"
    ).strip()
    if not subject or not org_id:
        raise IdentityError("Bearer token 缺少身份或租户声明")
    if visibility not in VISIBILITY_LEVELS:
        raise IdentityError("Bearer token 可见级别非法")
    roles = (
        frozenset(str(role) for role in roles_raw if role)
        if isinstance(roles_raw, list)
        else frozenset()
    )
    return Principal(subject=subject, org_id=org_id, roles=roles, visibility=visibility)


def _claim_at_path(claims: dict[str, Any], path: str) -> Any:
    current: Any = claims
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def request_principal(request: Request) -> Principal | None:
    return getattr(request.state, "principal", None)


def resolve_org(principal: Principal | None, requested_org: str | None) -> str:
    if principal is None:
        return requested_org or "default"
    if requested_org and requested_org != principal.org_id:
        raise HTTPException(status_code=403, detail="无权访问其他租户")
    return principal.org_id


def read_visibility(principal: Principal | None, requested: str = "public") -> str:
    return principal.visibility if principal is not None else requested


def may_read_visibility(principal: Principal | None, document_visibility: str) -> bool:
    if principal is None:
        return True
    document_level = VISIBILITY_LEVELS.get(document_visibility, 99)
    return document_level <= VISIBILITY_LEVELS[principal.visibility]


def write_visibility(principal: Principal | None, requested: str) -> str:
    if principal is None:
        return requested
    requested_level = VISIBILITY_LEVELS.get(requested)
    principal_level = VISIBILITY_LEVELS[principal.visibility]
    if requested_level is None or requested_level > principal_level:
        raise HTTPException(status_code=403, detail="无权设置该可见级别")
    return requested


def require_roles(principal: Principal | None, *allowed: str) -> None:
    if principal is not None and principal.roles.isdisjoint(allowed):
        raise HTTPException(status_code=403, detail="当前角色无权执行该操作")
