"""OpenTelemetry 可观测（阶段四）：trace + metrics 本地导出。

- trace：TracerProvider + BatchSpanProcessor(ConsoleSpanExporter)——span 输出到标准错误，
  uvicorn 日志文件可捕获；FastAPI 自动 instrumentation（请求级 span）+ 手动 span（检索/入库链路）。
- metrics：MeterProvider + ConsoleMetricReader（每 METRIC_INTERVAL_S 打印当前计数/直方图摘要）。
- 零外部依赖：不接 OTLP collector；后续需要 Jaeger/Tempo 时替换 ConsoleSpanExporter 即可。

用法：
    from app.core.otel import setup_otel, get_tracer
    setup_otel()                      # main.py lifespan 调用一次（幂等）
    with get_tracer("answer").start_as_current_span("search_plan") as span:
        span.set_attribute("question", q)
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import Any, Iterator

from opentelemetry import metrics, trace
from opentelemetry.metrics import get_meter_provider, set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import get_tracer_provider, set_tracer_provider

from app.core.config import get_settings

SERVICE_NAME = "newrag"
METRIC_INTERVAL_S = 60


@lru_cache
def _build_providers() -> bool:
    """构建 Tracer/Meter provider（幂等，仅首次执行；未启用时返回 False）。"""
    settings = get_settings()
    if not settings.otel_enabled:
        return False
    resource = Resource.create({"service.name": SERVICE_NAME, "service.version": "0.1.0"})
    tp = TracerProvider(resource=resource)
    tp.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    set_tracer_provider(tp)
    mp = MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(
                ConsoleMetricExporter(), export_interval_millis=METRIC_INTERVAL_S * 1000
            )
        ],
    )
    set_meter_provider(mp)
    return True


def setup_otel() -> bool:
    """应用启动时调用（main.py lifespan）。返回是否启用。"""
    return _build_providers()


def get_tracer(name: str) -> trace.Tracer:
    return get_tracer_provider().get_tracer(name)


def enabled() -> bool:
    return get_settings().otel_enabled


def get_meter(name: str) -> metrics.Meter:
    return get_meter_provider().get_meter(name)


@contextmanager
def make_span(
    name: str,
    tracer_name: str = SERVICE_NAME,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """手动 span 上下文管理器：with make_span("search_plan", attributes={...}) as span。
    未启用 OTel 时为空上下文（零开销），yield 的 span 可继续 set_attribute。"""
    if not enabled():
        yield None
        return
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                # OTel 不接受 None 值（如 year=None），统一跳过
                if v is None:
                    continue
                span.set_attribute(k, v)
        yield span
