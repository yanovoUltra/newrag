"""SQLAlchemy ORM 实体。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    """文档登记主表。"""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    file_type: Mapped[str] = mapped_column(String(16))  # pdf/docx/xlsx/png/jpg
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending/parsing/chunking/embedding/indexed/failed
    org_id: Mapped[str] = mapped_column(String(64), index=True)
    visibility: Mapped[str] = mapped_column(String(16), default="public")  # public/internal/restricted
    fiscal_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fiscal_quarter: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Task(Base):
    """解析任务状态（供前端轮询）。"""

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    type: Mapped[str] = mapped_column(String(16), default="ingest")  # ingest/delete
    doc_id: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running/success/failed
    stage: Mapped[str] = mapped_column(String(32), default="queued")  # queued/parse/chunk/embed/index
    progress: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(String(500), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class ChatRecord(Base):
    """问答记录（持久化）：离线复评 / RAGAS 评测的历史回答来源。"""

    __tablename__ = "chat_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    org_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    visibility: Mapped[str] = mapped_column(String(16), default="public")
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, default="")
    citations: Mapped[str] = mapped_column(Text, default="[]")  # JSON 序列化引用列表
    intent: Mapped[str] = mapped_column(String(32), default="")
    usage: Mapped[str] = mapped_column(String(500), default="{}")  # JSON：tokens 等
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class EvalResult(Base):
    """评测结果（RAGAS / 检索消融等）：指标 + 明细 JSON。"""

    __tablename__ = "eval_results"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    eval_name: Mapped[str] = mapped_column(String(64), index=True)  # retrieval_ablation / ragas / bm25_vs_sparse
    scope: Mapped[str] = mapped_column(String(255), default="")  # 数据集/文档范围
    metrics: Mapped[str] = mapped_column(Text, default="{}")  # JSON：{ndcg_at_8: 0.21, ...}
    payload: Mapped[str] = mapped_column(Text, default="{}")  # JSON：明细/报告
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class BenchmarkRun(Base):
    """竞品对标分析快照：保存输入与可审计结果，便于复看和导出。"""

    __tablename__ = "benchmark_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    org_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    visibility: Mapped[str] = mapped_column(String(16), default="public")
    name: Mapped[str] = mapped_column(String(120), default="")
    request_json: Mapped[str] = mapped_column(Text, default="{}")
    result_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class FinancialField(Base):
    """财务字段（结构化指标 → 独立索引）：支撑"某年某指标"类问答的精确取值。"""

    __tablename__ = "financial_fields"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    doc_id: Mapped[str] = mapped_column(String(32), index=True)
    org_id: Mapped[str] = mapped_column(String(64), index=True)
    visibility: Mapped[str] = mapped_column(String(16), default="public")
    metric: Mapped[str] = mapped_column(String(32), index=True)  # canonical key，如 net_profit
    metric_label: Mapped[str] = mapped_column(String(64), default="")
    year: Mapped[int] = mapped_column(Integer, index=True)
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(16), nullable=True)
    raw: Mapped[str] = mapped_column(String(255), default="")
    source_chunk_id: Mapped[str] = mapped_column(String(32), default="", index=True)
    company: Mapped[str] = mapped_column(String(64), default="", index=True)
    source: Mapped[str] = mapped_column(String(8), default="table")  # table | text
    page: Mapped[int] = mapped_column(Integer, default=0)
    section_path: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class NumberPhraseIndex(Base):
    """数字+单位短语倒排索引（离线构建，scripts/build_number_phrase_index.py）。

    通用 embedding 对精确数值不敏感（"1.2%" 与 "1.23%" 无法区分）——本表把每个
    chunk 的"数值+单位"短语（归一化后）映射到 chunk_id，查询侧精确匹配后硬插候选。
    """

    __tablename__ = "number_phrase_index"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    phrase: Mapped[str] = mapped_column(String(32), index=True)  # 归一化短语，如 "1.2%" / "15434家"
    chunk_id: Mapped[str] = mapped_column(String(32), index=True)
    doc_id: Mapped[str] = mapped_column(String(32), index=True)
    org_id: Mapped[str] = mapped_column(String(64), index=True)
    visibility: Mapped[str] = mapped_column(String(16), default="public")
    fiscal_year: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class RetrievalFallbackLog(Base):
    """分级降级兜底日志（§11）：记录每次触发与每级执行的召回健康度，供迭代优化词典/规则。"""

    __tablename__ = "retrieval_fallback_log"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    org_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    question: Mapped[str] = mapped_column(Text, index=True)
    level: Mapped[str] = mapped_column(String(16))  # L0_trigger / L1 / L2a / L2b / L3 / L4 / stop
    trigger_reason: Mapped[str] = mapped_column(String(64), default="")
    real_hits: Mapped[int] = mapped_column(Integer, default=0)
    candidate_count: Mapped[int] = mapped_column(Integer, default=0)
    params_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
