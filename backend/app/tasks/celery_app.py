"""Celery 应用（阶段四）：broker=Redis，任务状态沿用 SQLite（registry Task 表）。

Worker 启动（Windows 无 prefork，用 solo/threads 池）：
    cd backend
    .venv\\Scripts\\celery.exe -A app.tasks.celery_app worker -P solo --loglevel=info

API 侧通过 config `task_backend=celery` 启用（默认 background=FastAPI BackgroundTasks，
保证无 worker 环境/测试不受影响）。任务提交见 app/tasks/ingest_task.py。
"""

from __future__ import annotations

from celery import Celery
from celery.signals import worker_process_init

from app.core.config import get_settings
from app.core.otel import setup_otel
from app.store.registry import init_db

_settings = get_settings()

# worker 进程初始化 registry：模块级调用兜底 solo 单进程；worker_process_init 信号
# 覆盖 prefork 多进程场景（fork 后子进程需要独立会话工厂）。init_db 幂等，API 进程同样安全。
init_db()
# worker 进程启用 OTel（ingest.run 阶段 span 输出；未开启时零开销）
setup_otel()


@worker_process_init.connect
def _init_db_on_worker(**kwargs) -> None:
    init_db()


celery_app = Celery(
    "newrag",
    broker=_settings.redis_url,
    include=["app.tasks.ingest_task"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    # 单 worker 处理任务数上限后回收，防长驻内存泄漏（外部 API 客户端/嵌入缓存）
    worker_max_tasks_per_child=100,
    # Celery 5.6：broker 连接失败时在启动阶段自动重试（Redis 未就绪不再直接退出）
    broker_connection_retry_on_startup=True,
)
