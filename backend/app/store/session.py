"""会话记忆（可选）：按 session_id 存最近 N 轮问答，供多轮上下文注入。

- 存储：Redis（key=session:{id}，JSON 列表，TTL 1 天）；Redis 不可用时静默降级为无状态。
- 只读/只写接口均幂等，不抛异常，保证不阻断问答主流程。
"""

from __future__ import annotations

import json

from app.core.config import get_settings
from app.core.logging import get_logger
from app.store.redisx import get_redis

logger = get_logger(__name__)

SESSION_TTL = 86400  # 1 天


def _redis():
    return get_redis()


def _key(session_id: str) -> str:
    return f"session:{session_id}"


def get_history(session_id: str) -> list[dict]:
    """返回历史消息 [{role: user|assistant, content}]，无历史或异常返回空列表。"""
    if not session_id:
        return []
    r = _redis()
    if r is None:
        return []
    try:
        raw = r.get(_key(session_id))
        if not raw:
            return []
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("读取会话历史失败: %s", e)
        return []


def append_turn(session_id: str, question: str, answer: str) -> None:
    """追加一轮问答；按 session_window 裁剪历史（只保留最近 N 轮）。"""
    if not session_id:
        return
    r = _redis()
    if r is None:
        return
    try:
        history = get_history(session_id)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer or ""})
        window = get_settings().session_window
        if window > 0:
            history = history[-window * 2 :]
        r.set(_key(session_id), json.dumps(history, ensure_ascii=False), ex=SESSION_TTL)
    except Exception as e:
        logger.warning("写入会话历史失败: %s", e)
