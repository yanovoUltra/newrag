"""离线入库管线：解析 → (OCR) → 章节树 → 切分 → 向量化 → 入库，逐阶段落盘可续跑。"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger, set_trace_id
from app.core.otel import enabled, get_tracer
from app.embed.embedder import get_embedder
from app.parsers.base import LayoutResult, parse_layout
from app.parsers.ocr import ocr_page_images
from app.parsers.structure import Section, build_section_tree
from app.splitter.chunker import Chunk, build_leaves, build_parent_blocks, embedding_text_for
from app.splitter.semantic import refine_leaves_semantically
from app.store import qdrant as qdrant_store
from app.store.registry import (
    get_document,
    load_stage,
    save_stage,
    stage_done,
    update_document,
    update_task,
)

logger = get_logger(__name__)

VECTOR_BATCH_SIZE = 128


class PipelineError(Exception):
    pass


def _log_stage(doc_id: str, stage: str, t0: float, span=None) -> None:
    """阶段耗时埋点（JSON 日志带 trace_id=doc:{doc_id}，可串起整条入库链路；
    OTel 启用时同时记录 span attribute stage_{stage}_s，trace 内一眼看各阶段耗时）。"""
    dur = time.monotonic() - t0
    logger.info("doc %s stage %s done: %.1fs", doc_id, stage, dur)
    if span is not None:
        span.set_attribute(f"stage_{stage}_s", round(dur, 2))


def _update_task_progress(task_id: str, stage: str, progress: int, message: str = "") -> None:
    update_task(task_id, stage=stage, progress=progress, message=message)


def _check_deadline(deadline: float, doc_id: str) -> None:
    """任务级超时：超过 deadline 立即中止（协作式，由异常分支统一标记 failed）。"""
    if time.monotonic() > deadline:
        raise PipelineError(f"解析超时（>{get_settings().ingest_timeout}s），已中止")


def run_ingest(
    doc_id: str,
    file_path: Path,
    org_id: str,
    visibility: str,
    fiscal_year: int | None = None,
    fiscal_quarter: int | None = None,
    task_id: str | None = None,
) -> dict:
    """串行执行全流程；中间结果落盘 data/pipeline/{doc_id}/，可断点续跑。"""
    settings = get_settings()
    deadline = time.monotonic() + settings.ingest_timeout
    # 后台任务独立链路：trace_id 固定为 doc:{doc_id}，入库各阶段日志可串联
    set_trace_id(f"doc:{doc_id}")
    t_start = time.monotonic()
    span = None
    if enabled():
        span = get_tracer("ingest").start_span("ingest.run")
        span.set_attribute("doc_id", doc_id)
    try:
        # ---- 阶段 1：解析（版式层）----
        t0 = time.monotonic()
        if task_id:
            _update_task_progress(task_id, "parse", 15, "正在解析文档")
        update_document(doc_id, status="parsing")
        if not stage_done(doc_id, "layout"):
            layout = parse_layout(file_path)
            save_stage(doc_id, "layout", layout.to_dict())
        else:
            layout = load_layout(doc_id)  # 已反序列化为 LayoutResult
        _check_deadline(deadline, doc_id)

        # ---- 阶段 1.5：OCR 按需触发（提取率 < 80% 且启用）----
        if layout.text_extraction_rate < 0.8 and settings.ocr_enabled and file_path.suffix.lower() == ".pdf":
            update_document(doc_id, status="parsing")
            ocr_text = ocr_page_images(file_path, [p.page_no for p in layout.pages if p.scanned])
            if ocr_text:
                for page in layout.pages:
                    if page.scanned and page.page_no in ocr_text:
                        page.text = ocr_text[page.page_no]
                        page.scanned = False
                save_stage(doc_id, "layout", layout.to_dict())
            _check_deadline(deadline, doc_id)

        # ---- 阶段 1.6：数据清洗（页眉页脚去重 + NFKC 归一化，幂等）----
        if settings.clean_enable:
            from app.parsers.clean import clean_layout

            clean_layout(layout)
            save_stage(doc_id, "layout", layout.to_dict())
        _check_deadline(deadline, doc_id)
        _log_stage(doc_id, "parse", t0, span)

        # ---- 阶段 2：章节树（结构层）----
        t0 = time.monotonic()
        if task_id:
            _update_task_progress(task_id, "chunk", 40, "正在构建章节树")
        update_document(doc_id, status="chunking")
        if not stage_done(doc_id, "sections"):
            sections = build_section_tree(layout)
            save_stage(doc_id, "sections", {"sections": [s.to_dict() for s in sections]})
        else:
            sections = [Section.from_dict(s) for s in load_stage(doc_id, "sections")["sections"]]
        _check_deadline(deadline, doc_id)

        # ---- 阶段 2.5：字段抽取（结构化指标 → 独立索引，幂等 replace）----
        if settings.fields_enabled:
            from app.fields.extract import extract_fields
            from app.store.registry import replace_fields

            field_records = extract_fields(doc_id, layout, sections)
            n_fields = replace_fields(doc_id, field_records, org_id, visibility)
            logger.info("doc %s fields extracted: %d", doc_id, n_fields)
        _log_stage(doc_id, "sections", t0, span)

        # ---- 阶段 3：切分（Small-to-Big + 语义精切）----
        t0 = time.monotonic()
        if not stage_done(doc_id, "chunks"):
            # 3a) 结构叶子（章节/表格边界 + 目标 token 聚合）
            leaves = build_leaves(doc_id, layout, sections, doc_title=file_path.stem)
            # 3b) 语义精切：仅超长散文叶子按嵌入相似度断点细分（配置开启且非 mock 时生效）
            leaves = refine_leaves_semantically(leaves)
            # 3c) 章节级父块（叶子 parent_id 在此构建）
            chunks = build_parent_blocks(doc_id, leaves) + leaves
            save_stage(doc_id, "chunks", {"chunks": [c.to_dict() for c in chunks]})
        else:
            chunks = [Chunk.from_dict(c) for c in load_stage(doc_id, "chunks")["chunks"]]

        if not chunks:
            raise PipelineError("切分结果为空（文档无可检索内容）")
        _check_deadline(deadline, doc_id)
        _log_stage(doc_id, "chunks", t0, span)

        # ---- 阶段 3.5：字段 source_chunk_id 回填（relay 强候选前提）----
        # 依赖阶段 3 的叶子块（内容含 指标标签+数值 才能定位），故必须在切分后执行；
        # 幂等：只回填未定位字段，重传/续跑不重复修改。
        if settings.fields_enabled:
            try:
                from app.fields.backfill import backfill_doc_fields

                n_fill = backfill_doc_fields(doc_id, chunks)
                if n_fill:
                    logger.info("doc %s fields backfilled: %d", doc_id, n_fill)
            except Exception as e:  # 回填失败不阻断主流程（检索有 rerank 兜底）
                logger.warning("doc %s fields backfill 失败（跳过）: %s", doc_id, e)
        _check_deadline(deadline, doc_id)

        # ---- 阶段 3.6：LLM 提炼（章节父块摘要 + 文档综述，只做检索锚点）----
        # 摘要块 chunk_type=summary 入库（parent_id 指向原文父块）；检索命中由 search 层
        # 展开为原文父块，摘要本身不进生成上下文（防 LLM 编造数字）。幂等 stage summaries。
        if settings.summary_enabled and _summary_applies(doc_id):
            t0 = time.monotonic()
            summaries, doc_summary = _build_doc_summaries(
                doc_id, chunks, file_path.stem, task_id,
            )
            if summaries or doc_summary:
                chunks = _append_summary_chunks(doc_id, chunks, summaries, doc_summary)
            _log_stage(doc_id, "summaries", t0, span)

        # 解析/章节对象不再参与后续阶段，提前释放大型中间对象。
        layout = None
        sections = None
        leaves = None

        # ---- 阶段 4：向量化（稠密 + 稀疏）----
        t0 = time.monotonic()
        if task_id:
            _update_task_progress(task_id, "embed", 65, f"正在向量化 {len(chunks)} 个分块")
        update_document(doc_id, status="embedding")
        if not stage_done(doc_id, "vectors"):
            embedder = get_embedder()
            vector_stage = _embed_chunks_batched(
                doc_id,
                chunks,
                file_path.stem,
                embedder,
            )
        else:
            vector_stage = load_stage(doc_id, "vectors") or {}
        _check_deadline(deadline, doc_id)
        _log_stage(doc_id, "embed", t0, span)

        # ---- 阶段 5：入库 ----
        t0 = time.monotonic()
        if task_id:
            _update_task_progress(task_id, "index", 85, "正在写入向量库")
        doc = get_document(doc_id)
        qdrant_store.ensure_collection()
        n = 0
        for chunk_batch, vectors, sparse_vectors in _iter_vector_batches(
            doc_id, chunks, vector_stage
        ):
            if sparse_vectors:
                from app.retrieval.idf import apply_idf

                sparse_vectors = [apply_idf(sv) for sv in sparse_vectors]
            if len(vectors) != len(chunk_batch):
                raise PipelineError("向量数与分块数不一致，中止入库")
            n += qdrant_store.upsert_chunks(
                doc_id=doc_id,
                chunks=chunk_batch,
                dense_vectors=vectors,
                doc_name=doc.filename if doc else file_path.name,
                org_id=org_id,
                visibility=visibility,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
                sparse_vectors=sparse_vectors,
            )
            _check_deadline(deadline, doc_id)
        update_document(doc_id, status="indexed", chunk_count=n, error=None)
        if task_id:
            _update_task_progress(task_id, "index", 100, "入库完成")
            update_task(task_id, status="success", progress=100, message=f"解析完成，共 {n} 个分块")
        _log_stage(doc_id, "index", t0, span)
        logger.info("doc %s ingest total: %.1fs", doc_id, time.monotonic() - t_start)
        if span is not None:
            span.set_attribute("total_s", round(time.monotonic() - t_start, 2))
            span.end()
        return {"doc_id": doc_id, "chunks": n}

    except Exception as e:
        logger.exception("ingest failed for doc %s", doc_id)
        if span is not None:
            from opentelemetry.trace import Status, StatusCode

            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)[:200]))
            span.end()
        # 三写补偿：Qdrant 写入失败/半写时，回滚该文档已写入的向量点，
        # 避免"向量库有孤儿点、registry 却标记 failed"的不一致（文件/阶段产物保留供断点续跑）。
        try:
            qdrant_store.delete_doc(doc_id)
        except Exception as ce:
            logger.warning("向量回滚失败（可手工清理）: %s", ce)
        update_document(doc_id, status="failed", error=str(e))
        if task_id:
            update_task(task_id, status="failed", message=str(e))
        raise


def load_layout(doc_id: str) -> LayoutResult:
    data = load_stage(doc_id, "layout")
    if data is None:
        raise PipelineError("layout stage missing")
    return LayoutResult.from_dict(data)


def _embed_chunks_batched(doc_id: str, chunks: list[Chunk], doc_title: str, embedder) -> dict:
    """按有界批次向量化并落盘；最后写 manifest 作为阶段完成标记。"""
    batch_stages: list[str] = []
    for start in range(0, len(chunks), VECTOR_BATCH_SIZE):
        chunk_batch = chunks[start : start + VECTOR_BATCH_SIZE]
        texts = [embedding_text_for(c, doc_title=doc_title) for c in chunk_batch]
        vectors, sparse = embedder.embed_texts_with_sparse(texts, text_type="document")
        if len(vectors) != len(chunk_batch):
            raise PipelineError("向量数与分块数不一致，中止入库")
        stage_name = f"vectors_{start // VECTOR_BATCH_SIZE:05d}"
        save_stage(doc_id, stage_name, {"vectors": vectors, "sparse": sparse})
        batch_stages.append(stage_name)
    manifest = {
        "format": "batches-v1",
        "count": len(chunks),
        "batch_size": VECTOR_BATCH_SIZE,
        "stages": batch_stages,
    }
    save_stage(doc_id, "vectors", manifest)
    return manifest


def _iter_vector_batches(doc_id: str, chunks: list[Chunk], stage: dict):
    """读取新分批格式，并兼容存量单文件向量阶段。"""
    if stage.get("format") == "batches-v1":
        if int(stage.get("count", -1)) != len(chunks):
            raise PipelineError("向量数与分块数不一致，中止入库")
        offset = 0
        for stage_name in stage.get("stages", []):
            data = load_stage(doc_id, stage_name) or {}
            vectors = data.get("vectors") or []
            sparse = data.get("sparse")
            end = offset + len(vectors)
            yield chunks[offset:end], vectors, sparse
            offset = end
        if offset != len(chunks):
            raise PipelineError("向量批次不完整，中止入库")
        return

    vectors = stage.get("vectors") or []
    sparse = stage.get("sparse")
    for start in range(0, len(vectors), VECTOR_BATCH_SIZE):
        end = start + VECTOR_BATCH_SIZE
        yield chunks[start:end], vectors[start:end], sparse[start:end] if sparse else None


# ---------- 阶段 3.6：LLM 提炼（章节摘要 + 文档综述，只做检索锚点） ----------

def _summary_applies(doc_id: str) -> bool:
    """开关 + 按文档灰度：summary_doc_ids 为空 = 全部文档，否则仅白名单。"""
    ids = [x.strip() for x in get_settings().summary_doc_ids.split(",") if x.strip()]
    return not ids or doc_id in ids


def _build_doc_summaries(
    doc_id: str,
    chunks: list[Chunk],
    doc_title: str,
    task_id: str | None = None,
) -> tuple[list[tuple[str, str]], str]:
    """生成/复用章节摘要 + 文档综述（幂等 stage summaries，断点续跑不重复调 LLM）。"""
    from app.generation.summarize import summarize_doc_sections

    if not stage_done(doc_id, "summaries"):
        parents = [c for c in chunks if c.chunk_type == "section"]
        sections = [(c.id, c.content) for c in parents]
        chapters = "\n".join(f"- {c.section_path}" for c in parents[:80])
        if not sections:
            save_stage(doc_id, "summaries", {"sections": {}, "doc_summary": ""})
            return [], ""
        if task_id:
            _update_task_progress(task_id, "summaries", 55, f"正在提炼 {len(sections)} 个章节")
        ok, doc_summary = summarize_doc_sections(sections, doc_title, chapters)
        save_stage(doc_id, "summaries", {"sections": dict(ok), "doc_summary": doc_summary})
        return ok, doc_summary
    data = load_stage(doc_id, "summaries") or {}
    return list((data.get("sections") or {}).items()), data.get("doc_summary") or ""


def _append_summary_chunks(
    doc_id: str,
    chunks: list[Chunk],
    summaries: list[tuple[str, str]],
    doc_summary: str,
) -> list[Chunk]:
    """构造摘要块（chunk_type=summary）追加到 chunks：章节摘要 parent_id=原文父块，
    文档综述 parent_id=None（检索命中直接作为概览证据）。"""
    parent_by_id = {c.id: c for c in chunks if c.chunk_type == "section"}
    extra: list[Chunk] = []
    for pid, text in summaries:
        p = parent_by_id.get(pid)
        if not p:
            continue
        extra.append(Chunk(
            id=uuid.uuid4().hex, doc_id=doc_id, page=p.page,
            section_path=p.section_path, chunk_type="summary",
            content=text, token_count=max(1, len(text) // 2), parent_id=pid,
        ))
    if doc_summary:
        extra.append(Chunk(
            id=uuid.uuid4().hex, doc_id=doc_id, page=1,
            section_path="文档综述", chunk_type="summary",
            content=doc_summary, token_count=max(1, len(doc_summary) // 2),
            parent_id=None,
        ))
    if extra:
        logger.info("doc %s summaries appended: %d 块（摘要 %d + 综述 %d）",
                    doc_id, len(extra), len(summaries), 1 if doc_summary else 0)
    return chunks + extra
