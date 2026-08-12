"""结构化 JSON 日志：JSON 输出 + trace_id 链路追踪。

- 全部日志走 logger（禁止 print），统一 JSON 格式，便于采集与检索；
- trace_id 通过 ContextVar 注入（HTTP 中间件 / 后台任务入口设置），
  同一请求全链路日志共享同一 trace_id，可串起 上传→解析→检索→生成 各节点；
- 敏感信息（API 密钥等）由调用方保证不入日志（配置层不打印密钥）。
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

_CONFIGURED = False


def set_trace_id(tid: str) -> None:
    """设置当前上下文的 trace_id（请求中间件 / 后台任务入口调用）。"""
    trace_id_var.set(tid or "")


def get_trace_id() -> str:
    return trace_id_var.get()


class JsonFormatter(logging.Formatter):
    """一行一条 JSON：ts / level / logger / msg / trace_id；异常时附 exc_info。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": record.__dict__.get("trace_id", "") or get_trace_id(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
