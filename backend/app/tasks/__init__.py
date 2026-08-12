"""阶段四：Celery 异步任务（任务与 API 进程解耦，独立 worker 执行）。
任务状态沿用 SQLite registry Task 表（跨进程共享），不引入 result backend。"""
