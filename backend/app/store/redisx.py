"""Redis 连接（可选）：统一懒加载单例，不可用时返回 None（调用方静默降级）。"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client = None


def get_redis():
    """懒加载 Redis 客户端；不可用时返回 None。所有上层模块应容忍 None。"""
    global _client
    if _client is None:
        try:
            import redis

            settings = get_settings()
            r = redis.Redis.from_url(settings.redis_url, socket_timeout=2)
            r.ping()
            _client = r
        except Exception as e:
            logger.warning("Redis 不可用，相关缓存/会话功能降级: %s", e)
            _client = None
    return _client
