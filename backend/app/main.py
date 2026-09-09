"""FastAPI 应用入口。"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import benchmark, chat, config_public, documents, evaluations, metrics, tasks
from app.core.config import get_settings
from app.core.logging import get_logger, get_trace_id, set_trace_id, setup_logging
from app.core.otel import setup_otel
from app.core.security import AuthMiddleware
from app.store import qdrant as qdrant_store
from app.store.redisx import get_redis
from app.store.registry import init_db

logger = get_logger(__name__)

_health_lock = threading.Lock()
_health_cached_at = 0.0
_health_cached: dict[str, str] | None = None
_health_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="health")


def _dependency_health() -> dict[str, str]:
    """依赖检查最多等待 1 秒；复用已有 Qdrant/Redis 单例。"""
    def qdrant_check() -> None:
        qdrant_store.get_client().get_collections()

    def redis_check() -> None:
        client = get_redis()
        if client is None:
            raise RuntimeError("unavailable")
        client.ping()

    futures = {
        "qdrant": _health_executor.submit(qdrant_check),
        "redis": _health_executor.submit(redis_check),
    }
    checks = {"app": "ok"}
    for name, future in futures.items():
        try:
            future.result(timeout=1.0)
            checks[name] = "ok"
        except TimeoutError:
            checks[name] = "down(timeout)"
        except Exception as exc:  # noqa: BLE001
            checks[name] = "down(error)"
            logger.warning("dependency health failed: %s type=%s", name, type(exc).__name__)
    return checks


def _cached_dependency_health() -> dict[str, str]:
    global _health_cached_at, _health_cached
    now = time.monotonic()
    with _health_lock:
        if _health_cached is not None and now - _health_cached_at < 5.0:
            return dict(_health_cached)
        _health_cached = _dependency_health()
        _health_cached_at = time.monotonic()
        return dict(_health_cached)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    from app.core.runtime import validate_model_runtime

    validate_model_runtime(settings)
    settings.resolved_upload_dir.mkdir(parents=True, exist_ok=True)
    settings.resolved_pipeline_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    # 阶段四：OTel 可观测（本地导出；未启用时零开销）
    if setup_otel():
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
            logger.info("OTel enabled: FastAPI instrumentation installed")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "OTel instrumentation 失败（继续启动）: type=%s",
                type(exc).__name__,
            )
    try:
        qdrant_store.ensure_collection()
    except Exception as e:
        logger.warning("Qdrant 初始化失败，未修改现有集合: type=%s", type(e).__name__)
    logger.info("app startup done, env=%s", settings.app_env)
    yield


class RequestIdMiddleware:
    """每个 HTTP 请求注入 trace_id：日志上下文 + 响应头 X-Trace-Id，并记录请求总耗时。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        tid = uuid.uuid4().hex[:12]
        set_trace_id(tid)
        method = scope.get("method", "")
        start = time.perf_counter()
        logger.info("request start: method=%s", method)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-trace-id", tid.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            dur_ms = (time.perf_counter() - start) * 1000
            logger.info("request done: method=%s duration_ms=%.0f", method, dur_ms)
            set_trace_id("")


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
    # trace_id 最外层：认证/业务日志都能带上请求链路号
    app.add_middleware(RequestIdMiddleware)

    # 全局异常兜底：未捕获异常统一 500 JSON（详情只进日志，不回给客户端），
    # 入参校验失败统一 422（格式与 FastAPI 默认一致，前端无需改动）
    @app.exception_handler(RequestValidationError)
    async def validation_exc_handler(request: Request, exc: RequestValidationError):
        errors = [
            {
                "type": item.get("type", "validation_error"),
                "loc": item.get("loc", ()),
                "msg": item.get("msg", "输入无效"),
            }
            for item in exc.errors()
        ]
        logger.warning(
            "validation error: method=%s count=%d",
            request.method,
            len(errors),
        )
        return JSONResponse(
            status_code=422,
            content={
                "detail": jsonable_encoder(errors),
                "code": "invalid_request",
                "request_id": get_trace_id(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exc_handler(request: Request, exc: Exception):
        logger.error(
            "unhandled error: method=%s type=%s request_id=%s",
            request.method,
            type(exc).__name__,
            get_trace_id(),
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "服务器内部错误，请稍后重试",
                "code": "internal_error",
                "request_id": get_trace_id(),
            },
        )

    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(tasks.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(config_public.router, prefix="/api/v1")
    app.include_router(metrics.router, prefix="/api/v1")
    app.include_router(benchmark.router, prefix="/api/v1")
    app.include_router(evaluations.router, prefix="/api/v1")

    @app.get("/livez")
    def livez():
        return {"app": "ok"}

    @app.get("/healthz")
    def healthz():
        return _cached_dependency_health()

    return app


app = create_app()
