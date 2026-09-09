"""前端可安全读取的公开运行时配置。"""

from fastapi import APIRouter

from app.api.v1.documents import MIN_FISCAL_YEAR, _current_year
from app.core.config import get_settings
from app.core.runtime import (
    effective_runtime_mode,
    effective_runtime_profile,
    model_feature_status,
    model_service_status,
)
from app.parsers.base import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/public")
def public_config() -> dict:
    settings = get_settings()
    features = model_feature_status(settings)
    services = model_service_status(settings, probe=settings.runtime_endpoint_probe)
    return {
        "max_upload_mb": settings.max_upload_mb,
        "accepted_extensions": sorted(SUPPORTED_EXTENSIONS),
        "fiscal_year_min": MIN_FISCAL_YEAR,
        "current_year": _current_year(),
        "runtime_mode": effective_runtime_mode(settings, features),
        "runtime_profile": effective_runtime_profile(settings, features),
        "model_features": features,
        "model_services": services,
        "auth": {
            "mode": settings.auth_mode,
            "issuer": settings.oidc_issuer if settings.auth_mode == "oidc" else "",
            "client_id": (
                settings.oidc_frontend_client_id if settings.auth_mode == "oidc" else ""
            ),
        },
    }
