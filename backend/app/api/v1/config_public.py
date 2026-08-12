"""前端可安全读取的公开运行时配置。"""

from fastapi import APIRouter

from app.api.v1.documents import MIN_FISCAL_YEAR, _current_year
from app.core.config import get_settings
from app.parsers.base import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/public")
def public_config() -> dict:
    settings = get_settings()
    return {
        "max_upload_mb": settings.max_upload_mb,
        "accepted_extensions": sorted(SUPPORTED_EXTENSIONS),
        "fiscal_year_min": MIN_FISCAL_YEAR,
        "current_year": _current_year(),
    }
