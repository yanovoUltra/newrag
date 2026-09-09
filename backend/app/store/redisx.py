"""Redis 连接（可选）：统一懒加载单例，不可用时返回 None（调用方静默降级）。"""

from __future__ import annotations

import threading
import time

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client = None
_retry_after = 0.0
_connect_lock = threading.Lock()

_CONNECT_TIMEOUT_SECONDS = 0.25
_SOCKET_TIMEOUT_SECONDS = 1.0
_RETRY_INTERVAL_SECONDS = 5.0


def get_redis():
    """懒加载 Redis 客户端；不可用时返回 None。所有上层模块应容忍 None。"""
    global _client, _retry_after
    if _client is not None:
        return _client
    if time.monotonic() < _retry_after:
        return None
    with _connect_lock:
        if _client is not None:
            return _client
        if time.monotonic() < _retry_after:
            return None
        try:
            import redis

            settings = get_settings()
            cache_url = settings.redis_cache_url or settings.redis_url
            r = redis.Redis.from_url(
                cache_url,
                socket_connect_timeout=_CONNECT_TIMEOUT_SECONDS,
                socket_timeout=_SOCKET_TIMEOUT_SECONDS,
            )
            r.ping()
            _client = r
            _retry_after = 0.0
        except Exception as exc:
            logger.warning(
                "Redis 不可用，相关缓存/会话功能降级: type=%s",
                type(exc).__name__,
            )
            _client = None
            _retry_after = time.monotonic() + _RETRY_INTERVAL_SECONDS
    return _client
