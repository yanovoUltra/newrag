"""外部 API 调用指标（阶段四 OTel metrics）：计数 + 失败计数 + 延迟直方图。

数据经 ConsoleMetricReader 每 60s 输出到标准错误（uvicorn 日志可捕获）——
外部 API（嵌入/LLM/rerank）是系统最大瓶颈，调用量/延迟/错误率是核心可观测指标。
未启用 OTel 时零开销（no-op）。
"""

from __future__ import annotations

from app.core.otel import enabled, get_meter

_inited = False
_meter = None
_counter = None
_errors = None
_hist = None


def _ensure() -> bool:
    global _inited, _meter, _counter, _errors, _hist
    if _inited:
        return _meter is not None
    _inited = True
    if not enabled():
        return False
    _meter = get_meter("newrag.api")
    _counter = _meter.create_counter("api.calls", unit="1", description="外部 API 调用次数")
    _errors = _meter.create_counter("api.errors", unit="1", description="外部 API 调用失败次数")
    _hist = _meter.create_histogram("api.duration", unit="s", description="外部 API 调用耗时")
    return True


def record_api_call(api: str, duration_s: float, error: bool = False) -> None:
    """记录一次外部 API 调用（api ∈ embed/llm/rerank）。"""
    if not _ensure():
        return
    attrs = {"api": api}
    _counter.add(1, attrs)
    _hist.record(duration_s, attrs)
    if error:
        _errors.add(1, attrs)
