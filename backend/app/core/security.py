"""API 认证：HMAC 签名验签 + 时间戳窗口 + nonce 防重放。

协议（请求头）：
  X-Client-Key  : 客户端标识（明文）
  X-Timestamp   : Unix 秒（需落在 auth_timestamp_window 窗口内）
  X-Nonce       : 一次性随机串（同窗口内重复使用直接拒绝）
  X-Sign        : HMAC-SHA256(secret, 签名串).hexdigest()

签名串 = "{key}\\n{timestamp}\\n{nonce}\\n{METHOD}\\n{path}\\n{body_sha256}"
  - path 为原始路径（含 query，不含 host）；body_sha256 为请求体原文的 sha256。

nonce 去重：优先 Redis SETNX（跨 worker 生效）；Redis 不可用退化为进程内 LRU（单 worker 兜底）。
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections import OrderedDict

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_HEADERS = ("x-client-key", "x-timestamp", "x-nonce", "x-sign")

# ---- 进程内 nonce 兜底（Redis 不可用时） ----
_inmem_lock = threading.Lock()
_inmem_nonces: OrderedDict[str, float] = OrderedDict()
_INMEM_MAX = 10000


def sign_headers(
    method: str,
    path: str,
    body: bytes = b"",
    client_key: str | None = None,
    secret: str | None = None,
    timestamp: int | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    """客户端签名工具：生成需携带的四个请求头（测试/脚本/前端共用）。"""
    settings = get_settings()
    key = client_key or settings.auth_client_key
    sec = secret or settings.auth_secret
    if not key or not sec:
        raise ValueError("auth_client_key / auth_secret 未配置")
    ts = timestamp if timestamp is not None else int(time.time())
    n = nonce or hashlib.sha256(f"{ts}:{key}:{time.time()}:{id(object())}".encode()).hexdigest()[:32]
    sig_str = _signature_string(key, ts, n, method.upper(), path, body)
    return {
        "X-Client-Key": key,
        "X-Timestamp": str(ts),
        "X-Nonce": n,
        "X-Sign": hmac.new(sec.encode(), sig_str.encode(), hashlib.sha256).hexdigest(),
    }


def _signature_string(key: str, ts: int, nonce: str, method: str, path: str, body: bytes) -> str:
    body_sha = hashlib.sha256(body).hexdigest()
    return f"{key}\n{ts}\n{nonce}\n{method}\n{path}\n{body_sha}"


def _nonce_seen(key: str, nonce: str) -> bool:
    """nonce 去重：返回 True 表示已见过（重放）。优先 Redis SETNX。"""
    settings = get_settings()
    try:
        from app.store.redisx import get_redis

        r = get_redis()
        if r is not None:
            return not r.set(f"auth:nonce:{key}:{nonce}", 1, ex=settings.auth_nonce_ttl, nx=True)
    except Exception as e:
        logger.warning("nonce 去重降级到进程内: %s", e)
    # 进程内兜底
    now = time.time()
    mem_key = f"{key}:{nonce}"
    with _inmem_lock:
        if mem_key in _inmem_nonces:
            return True
        _inmem_nonces[mem_key] = now
        while len(_inmem_nonces) > _INMEM_MAX:
            _inmem_nonces.popitem(last=False)
        _inmem_nonces = OrderedDict((k, v) for k, v in _inmem_nonces.items() if now - v <= settings.auth_nonce_ttl)
        return False


def verify_request(method: str, path: str, body: bytes, headers: dict[str, str]) -> tuple[bool, str]:
    """服务端验签。返回 (是否通过, 拒绝原因)。"""
    settings = get_settings()
    missing = [h for h in _HEADERS if not headers.get(h)]
    if missing:
        return False, f"缺少认证头: {', '.join(missing)}"
    key = headers["x-client-key"]
    if key != settings.auth_client_key:
        return False, "client_key 无效"
    try:
        ts = int(headers["x-timestamp"])
    except ValueError:
        return False, "timestamp 非法"
    now = int(time.time())
    if abs(now - ts) > settings.auth_timestamp_window:
        return False, f"timestamp 超出窗口（偏差 {abs(now - ts)}s）"
    sig_str = _signature_string(key, ts, headers["x-nonce"], method.upper(), path, body)
    expect = hmac.new(settings.auth_secret.encode(), sig_str.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expect, headers["x-sign"]):
        return False, "签名不匹配（参数被篡改或密钥错误）"
    if _nonce_seen(key, headers["x-nonce"]):
        return False, "nonce 已使用（重放攻击）"
    return True, ""


class AuthMiddleware:
    """FastAPI 中间件：开启 AUTH_ENABLED 后保护 /api/v1/*。

    放行：/healthz、/livez、/api/v1/config/public、OpenAPI 文档、CORS 预检。
    未开启或未配置密钥时完全放行（默认开发行为，不影响既有流程）。
    """

    def __init__(self, app, *, protected_prefix: str = "/api/v1"):
        self.app = app
        self.protected_prefix = protected_prefix

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        settings = get_settings()
        if not settings.auth_enabled or not settings.auth_client_key or not settings.auth_secret:
            await self.app(scope, receive, send)
            return
        request = _Request(scope, receive, send)
        method = request.method
        path = request.path
        # 仅保护 /api/v1/*；其余（/healthz、OpenAPI 文档）与 CORS 预检（OPTIONS）放行
        if (
            method == "OPTIONS"
            or path == "/api/v1/config/public"
            or not path.startswith(self.protected_prefix)
        ):
            await self.app(scope, receive, send)
            return
        body = await request.body()
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        ok, reason = verify_request(method, path, body, headers)
        if not ok:
            response = _JSONResponse(401, {"detail": f"认证失败: {reason}"})
            await response(scope, receive, send)
            return
        # 验签通过：验签已消费请求体，包装 receive 将 body 重放给下游（下游只需读一次）
        sent = False

        async def replay_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)


class _Request:
    """极简请求封装：读取 method/path/body（缓存 body 供下游复用）。"""

    def __init__(self, scope, receive, send):
        self.scope = scope
        self._receive = receive
        self.method = scope.get("method", "")
        self.path = scope.get("path", "")
        self._body: bytes | None = None

    async def body(self) -> bytes:
        if self._body is None:
            chunks = []
            while True:
                msg = await self._receive()
                chunks.append(msg.get("body", b""))
                if not msg.get("more_body"):
                    break
            self._body = b"".join(chunks)
        return self._body


class _JSONResponse:
    def __init__(self, status: int, payload: dict):
        import json

        self.status = status
        self.body = json.dumps(payload, ensure_ascii=False).encode()
        self.headers = [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"content-length", str(len(self.body)).encode()),
        ]

    async def __call__(self, scope, receive, send):
        await send({
            "type": "http.response.start",
            "status": self.status,
            "headers": self.headers,
        })
        await send({"type": "http.response.body", "body": self.body})
