"""FastAPI 应用入口。"""

from __future__ import annotations

import redis
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

from app.api.v1 import chat, documents, tasks
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.security import AuthMiddleware
from app.store import qdrant as qdrant_store
from app.store.registry import init_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    settings.resolved_upload_dir.mkdir(parents=True, exist_ok=True)
    settings.resolved_pipeline_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    try:
        qdrant_store.ensure_collection()
    except Exception as e:
        logger.warning("Qdrant 初始化失败（稍后可通过重试/上传触发重建）: %s", e)
    logger.info("app startup done, env=%s", settings.app_env)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="多模态财报深度分析 RAG 系统", version="0.1.0", lifespan=lifespan)
    # 认证中间件在内（仅保护 /api/v1/*，默认关闭），CORS 在外（保证 401 也带 CORS 头）
    app.add_middleware(AuthMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(tasks.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")

    @app.get("/healthz")
    def healthz():
        settings = get_settings()
        checks: dict[str, str] = {"app": "ok"}
        try:
            QdrantClient(url=settings.qdrant_url, timeout=3).get_collections()
            checks["qdrant"] = "ok"
        except Exception as e:
            checks["qdrant"] = f"down({e})"
        try:
            r = redis.Redis.from_url(settings.redis_url, socket_timeout=3)
            r.ping()
            checks["redis"] = "ok"
        except Exception as e:
            checks["redis"] = f"down({e})"
        return checks

    return app


app = create_app()
