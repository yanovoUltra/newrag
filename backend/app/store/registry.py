"""SQLite 登记库：文档/任务 CRUD + 串行持久化阶段文件。"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import create_engine, delete, event, func, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.core.logging import get_logger
from app.fields.extract import FieldRecord
from app.models.entities import (
    Base,
    BenchmarkRun,
    ChatRecord,
    Document,
    EvalResult,
    FinancialField,
    NumberPhraseIndex,
    RetrievalFallbackLog,
    Task,
)

logger = get_logger(__name__)

_engine = None
_session_factory = None


def init_db() -> None:
    global _engine, _session_factory
    settings = get_settings()
    db_path = settings.resolved_registry_db
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

    @event.listens_for(_engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    Base.metadata.create_all(_engine)
    _migrate_fields_company()
    _ensure_indexes()
    logger.info("SQLite registry ready: %s", db_path)


def _migrate_fields_company() -> None:
    """轻量迁移：为已有 financial_fields 表补 company / source_chunk_id 列（新增列 not null 默认空串）。"""
    from sqlalchemy import text

    try:
        with _engine.begin() as conn:
            cols = [row[1] for row in conn.execute(text("PRAGMA table_info(financial_fields)"))]
            if "company" not in cols:
                conn.execute(text("ALTER TABLE financial_fields ADD COLUMN company VARCHAR(64) DEFAULT ''"))
                logger.info("financial_fields: 已新增 company 列（待回填）")
            if "source_chunk_id" not in cols:
                conn.execute(text("ALTER TABLE financial_fields ADD COLUMN source_chunk_id VARCHAR(32) DEFAULT ''"))
                logger.info("financial_fields: 已新增 source_chunk_id 列（待回填）")
    except Exception as e:  # noqa: BLE001
        logger.warning("financial_fields 列迁移跳过: %s", e)


def _ensure_indexes() -> None:
    """旧 SQLite 数据库补齐直查所需组合索引（幂等）。"""
    statements = (
        "CREATE INDEX IF NOT EXISTS ix_documents_org_sha256 ON documents (org_id, sha256)",
        "CREATE INDEX IF NOT EXISTS ix_documents_org_filename ON documents (org_id, filename)",
        "CREATE INDEX IF NOT EXISTS ix_financial_fields_lookup ON financial_fields "
        "(org_id, visibility, company, metric, year)",
        "CREATE INDEX IF NOT EXISTS ix_number_phrase_lookup ON number_phrase_index "
        "(org_id, visibility, phrase, fiscal_year)",
        "CREATE INDEX IF NOT EXISTS ix_benchmark_runs_org_created ON benchmark_runs "
        "(org_id, created_at)",
    )
    with _engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


def get_session() -> Session:
    if _session_factory is None:
        raise RuntimeError("registry not initialized, call init_db() first")
    return _session_factory()


# ---------- Document ----------

def create_document(
    filename: str,
    file_type: str,
    org_id: str,
    visibility: str,
    size_bytes: int,
    sha256: str = "",
    fiscal_year: int | None = None,
    fiscal_quarter: int | None = None,
) -> Document:
    with get_session() as s:
        doc = Document(
            filename=filename,
            file_type=file_type,
            org_id=org_id,
            visibility=visibility,
            size_bytes=size_bytes,
            sha256=sha256,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
        )
        s.add(doc)
        s.commit()
        s.refresh(doc)
        return doc


def get_document(doc_id: str) -> Document | None:
    with get_session() as s:
        return s.get(Document, doc_id)


def find_document_by_sha256(org_id: str, sha256: str) -> Document | None:
    with get_session() as s:
        return s.scalar(
            select(Document)
            .where(
                Document.org_id == org_id,
                Document.sha256 == sha256,
                Document.status != "archived",
            )
            .order_by(Document.created_at.desc())
            .limit(1)
        )


def find_document_by_filename(org_id: str, filename: str) -> Document | None:
    with get_session() as s:
        return s.scalar(
            select(Document)
            .where(
                Document.org_id == org_id,
                Document.filename == filename,
                Document.status != "archived",
            )
            .order_by(Document.created_at.desc())
            .limit(1)
        )


def list_documents(org_id: str | None = None, limit: int = 50, offset: int = 0) -> list[Document]:
    docs, _ = query_documents_page(
        org_id=org_id,
        limit=limit,
        offset=offset,
        include_archived=True,
    )
    return docs


def query_documents_page(
    org_id: str | None = None,
    filename: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    include_archived: bool = False,
) -> tuple[list[Document], int]:
    filters = []
    if org_id:
        filters.append(Document.org_id == org_id)
    if filename:
        filters.append(Document.filename.ilike(f"%{filename.strip()}%"))
    if status:
        filters.append(Document.status == status)
    if not include_archived:
        filters.append(Document.status != "archived")

    with get_session() as s:
        stmt = select(Document).where(*filters).order_by(Document.created_at.desc())
        count_stmt = select(func.count()).select_from(Document).where(*filters)
        docs = list(s.scalars(stmt.limit(limit).offset(offset)).all())
        total = int(s.scalar(count_stmt) or 0)
        return docs, total


def update_document(doc_id: str, **fields) -> None:
    with get_session() as s:
        doc = s.get(Document, doc_id)
        if doc is None:
            return
        for k, v in fields.items():
            setattr(doc, k, v)
        s.commit()


def delete_document(doc_id: str) -> None:
    with get_session() as s:
        doc = s.get(Document, doc_id)
        if doc:
            s.delete(doc)
            s.commit()


# ---------- Task ----------

def create_task(doc_id: str, type_: str = "ingest") -> Task:
    with get_session() as s:
        t = Task(doc_id=doc_id, type=type_)
        s.add(t)
        s.commit()
        s.refresh(t)
        return t


def get_task(task_id: str) -> Task | None:
    with get_session() as s:
        return s.get(Task, task_id)


def update_task(task_id: str, **fields) -> None:
    with get_session() as s:
        t = s.get(Task, task_id)
        if t is None:
            return
        for k, v in fields.items():
            setattr(t, k, v)
        s.commit()


# ---------- ChatRecord（问答记录） ----------

def create_chat_record(
    question: str,
    answer: str = "",
    session_id: str = "",
    org_id: str = "default",
    visibility: str = "public",
    citations: list | None = None,
    intent: str = "",
    usage: dict | None = None,
) -> ChatRecord:
    with get_session() as s:
        rec = ChatRecord(
            question=question,
            answer=answer,
            session_id=session_id,
            org_id=org_id,
            visibility=visibility,
            citations=json.dumps(citations or [], ensure_ascii=False),
            intent=intent,
            usage=json.dumps(usage or {}, ensure_ascii=False),
        )
        s.add(rec)
        s.commit()
        s.refresh(rec)
        return rec


def list_chat_records(org_id: str | None = None, limit: int = 200, offset: int = 0) -> list[ChatRecord]:
    with get_session() as s:
        stmt = select(ChatRecord).order_by(ChatRecord.created_at.desc()).limit(limit).offset(offset)
        if org_id:
            stmt = stmt.where(ChatRecord.org_id == org_id)
        return list(s.scalars(stmt).all())


# ---------- FinancialField（财务字段独立索引） ----------

def replace_fields(doc_id: str, records: list[FieldRecord], org_id: str, visibility: str) -> int:
    """整文档替换字段索引：先删该文档旧字段，再写入新抽取结果。"""
    with get_session() as s:
        s.execute(delete(FinancialField).where(FinancialField.doc_id == doc_id))
        for r in records:
            s.add(
                FinancialField(
                    doc_id=r.doc_id,
                    org_id=org_id,
                    visibility=visibility,
                    metric=r.metric,
                    metric_label=r.metric_label,
                    year=r.year,
                    value=r.value,
                    unit=r.unit,
                    raw=r.raw,
                    source_chunk_id=getattr(r, "source_chunk_id", ""),
                    company=r.company,
                    source=r.source,
                    page=r.page,
                    section_path=r.section_path,
                )
            )
        s.commit()
        return len(records)


def delete_fields(doc_id: str) -> None:
    with get_session() as s:
        s.execute(delete(FinancialField).where(FinancialField.doc_id == doc_id))
        s.commit()


def query_fields(
    metric_keys: list[str],
    org_id: str,
    user_visibility: str,
    year: int | None = None,
    company: str | None = None,
    limit: int = 20,
) -> list[FinancialField]:
    """按指标（+可选年份/公司）+ 权限过滤查询字段索引。company 用于主体公司过滤。"""
    from app.store.qdrant import visible_levels

    with get_session() as s:
        stmt = (
            select(FinancialField)
            .where(
                FinancialField.metric.in_(metric_keys),
                FinancialField.org_id == org_id,
                FinancialField.visibility.in_(visible_levels(user_visibility)),
            )
            .order_by(FinancialField.year.desc())
            .limit(limit)
        )
        if year is not None:
            stmt = stmt.where(FinancialField.year == year)
        if company:
            stmt = stmt.where(FinancialField.company == company)
        return list(s.scalars(stmt).all())


def list_field_companies(org_id: str, user_visibility: str) -> list[str]:
    """返回该 org 下字段索引中的去重公司名（用于问题主体匹配）。"""
    from app.store.qdrant import visible_levels

    with get_session() as s:
        rows = s.scalars(
            select(FinancialField.company)
            .where(
                FinancialField.org_id == org_id,
                FinancialField.visibility.in_(visible_levels(user_visibility)),
            )
            .distinct()
        ).all()
    return [c for c in rows if c]


def benchmark_dimensions(
    org_id: str, user_visibility: str
) -> tuple[list[str], list[int], dict[str, int]]:
    """返回对标工作台所需的公司、年份和各指标可用记录数。"""
    from app.store.qdrant import visible_levels

    filters = (
        FinancialField.org_id == org_id,
        FinancialField.visibility.in_(visible_levels(user_visibility)),
    )
    with get_session() as s:
        companies = list(
            s.scalars(
                select(FinancialField.company)
                .where(*filters, FinancialField.company != "")
                .distinct()
                .order_by(FinancialField.company)
            ).all()
        )
        years = list(
            s.scalars(
                select(FinancialField.year)
                .where(*filters)
                .distinct()
                .order_by(FinancialField.year.desc())
            ).all()
        )
        metric_rows = s.execute(
            select(FinancialField.metric, func.count(FinancialField.id))
            .where(*filters)
            .group_by(FinancialField.metric)
        ).all()
    return companies, years, {str(key): int(count) for key, count in metric_rows}


def query_benchmark_fields(
    metric_keys: list[str],
    companies: list[str],
    years: list[int],
    org_id: str,
    user_visibility: str,
    limit: int = 2000,
) -> list[FinancialField]:
    """一次查询完整对标矩阵，避免按公司/指标/年份产生 N+1 查询。"""
    from app.store.qdrant import visible_levels

    if not metric_keys or not companies or not years:
        return []
    with get_session() as s:
        stmt = (
            select(FinancialField)
            .where(
                FinancialField.metric.in_(metric_keys),
                FinancialField.company.in_(companies),
                FinancialField.year.in_(years),
                FinancialField.org_id == org_id,
                FinancialField.visibility.in_(visible_levels(user_visibility)),
            )
            .order_by(
                FinancialField.metric,
                FinancialField.company,
                FinancialField.year,
                FinancialField.created_at.desc(),
            )
            .limit(limit)
        )
        return list(s.scalars(stmt).all())


def get_documents_by_ids(doc_ids: list[str]) -> dict[str, Document]:
    """批量读取证据文档，供对标结果补齐文件名。"""
    if not doc_ids:
        return {}
    with get_session() as s:
        docs = s.scalars(select(Document).where(Document.id.in_(set(doc_ids)))).all()
        return {doc.id: doc for doc in docs}


# ---------- BenchmarkRun（竞品对标分析快照） ----------

def save_benchmark_run(
    run_id: str,
    org_id: str,
    visibility: str,
    name: str,
    request_data: dict,
    result_data: dict,
) -> BenchmarkRun:
    with get_session() as s:
        run = BenchmarkRun(
            id=run_id,
            org_id=org_id,
            visibility=visibility,
            name=name,
            request_json=json.dumps(request_data, ensure_ascii=False),
            result_json=json.dumps(result_data, ensure_ascii=False),
        )
        s.add(run)
        s.commit()
        s.refresh(run)
        return run


def get_benchmark_run(
    run_id: str, org_id: str, user_visibility: str
) -> BenchmarkRun | None:
    from app.store.qdrant import visible_levels

    with get_session() as s:
        return s.scalar(
            select(BenchmarkRun).where(
                BenchmarkRun.id == run_id,
                BenchmarkRun.org_id == org_id,
                BenchmarkRun.visibility.in_(visible_levels(user_visibility)),
            )
        )


def list_benchmark_runs(
    org_id: str, user_visibility: str, limit: int = 20
) -> list[BenchmarkRun]:
    from app.store.qdrant import visible_levels

    with get_session() as s:
        return list(
            s.scalars(
                select(BenchmarkRun)
                .where(
                    BenchmarkRun.org_id == org_id,
                    BenchmarkRun.visibility.in_(visible_levels(user_visibility)),
                )
                .order_by(BenchmarkRun.created_at.desc())
                .limit(limit)
            ).all()
        )


# ---------- NumberPhraseIndex（数字+单位短语倒排索引，离线构建） ----------

def replace_number_phrase_index(rows: list[tuple]) -> int:
    """整表重建数字短语倒排索引：先删全表，再写入 (phrase, chunk_id, doc_id, org_id, visibility, fiscal_year)。"""
    with get_session() as s:
        s.execute(delete(NumberPhraseIndex))
        for phrase, chunk_id, doc_id, org_id, visibility, fiscal_year in rows:
            s.add(
                NumberPhraseIndex(
                    phrase=phrase,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    org_id=org_id,
                    visibility=visibility,
                    fiscal_year=fiscal_year,
                )
            )
        s.commit()
        return len(rows)


def query_number_phrase_index(
    phrases: list[str],
    org_id: str,
    user_visibility: str,
    fiscal_year: int | None = None,
    doc_ids: list[str] | None = None,
    limit: int = 50,
) -> list[tuple[str, str]]:
    """数字短语精确查询：短语 → [(chunk_id, doc_id)]，按权限/年份/文档过滤。"""
    from app.store.qdrant import visible_levels

    if not phrases:
        return []
    with get_session() as s:
        stmt = (
            select(NumberPhraseIndex.chunk_id, NumberPhraseIndex.doc_id)
            .where(
                NumberPhraseIndex.phrase.in_(list(phrases)),
                NumberPhraseIndex.org_id == org_id,
                NumberPhraseIndex.visibility.in_(visible_levels(user_visibility)),
            )
            .distinct()
            .limit(limit)
        )
        if fiscal_year is not None:
            stmt = stmt.where(NumberPhraseIndex.fiscal_year == fiscal_year)
        if doc_ids:
            stmt = stmt.where(NumberPhraseIndex.doc_id.in_(doc_ids))
        return list(s.execute(stmt).all())


# ---------- EvalResult（评测结果） ----------

def save_eval_result(eval_name: str, scope: str, metrics: dict, payload: dict | None = None) -> EvalResult:
    with get_session() as s:
        rec = EvalResult(
            eval_name=eval_name,
            scope=scope,
            metrics=json.dumps(metrics, ensure_ascii=False),
            payload=json.dumps(payload or {}, ensure_ascii=False),
        )
        s.add(rec)
        s.commit()
        s.refresh(rec)
        return rec


# ---------- RetrievalFallbackLog（分级降级日志） ----------

def log_fallback(
    question: str,
    level: str,
    trigger_reason: str,
    real_hits: int,
    candidate_count: int,
    params_json: str = "{}",
    org_id: str = "default",
) -> None:
    """记录一次降级触发/每级执行的召回健康度（§11，供迭代优化词典与规则）。"""
    try:
        with get_session() as s:
            s.add(
                RetrievalFallbackLog(
                    org_id=org_id,
                    question=(question or "")[:500],
                    level=level,
                    trigger_reason=(trigger_reason or "")[:64],
                    real_hits=int(real_hits),
                    candidate_count=int(candidate_count),
                    params_json=params_json[:2000],
                )
            )
            s.commit()
    except Exception as e:  # noqa: BLE001  日志失败不影响检索主流程
        logger.warning("降级日志写入失败（忽略）: %s", e)


# ---------- 串行持久化（断点续跑） ----------

def pipeline_dir(doc_id: str) -> Path:
    return get_settings().resolved_pipeline_dir / doc_id


def save_stage(doc_id: str, stage: str, data: dict) -> Path:
    """落盘阶段中间结果，如 01_layout.json / 04_chunks.json。"""
    p = pipeline_dir(doc_id)
    p.mkdir(parents=True, exist_ok=True)
    idx = {"layout": "01_layout", "ocr": "02_ocr", "sections": "03_sections",
           "chunks": "04_chunks", "vectors": "05_vectors"}.get(stage, stage)
    path = p / f"{idx}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    logger.info("[%s] stage saved: %s", doc_id, path.name)
    return path


def load_stage(doc_id: str, stage: str) -> dict | None:
    p = pipeline_dir(doc_id)
    idx = {"layout": "01_layout", "ocr": "02_ocr", "sections": "03_sections",
           "chunks": "04_chunks", "vectors": "05_vectors"}.get(stage, stage)
    path = p / f"{idx}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stage_done(doc_id: str, stage: str) -> bool:
    return load_stage(doc_id, stage) is not None
