"""离线入库管线：解析 → (OCR) → 章节树 → 切分 → 向量化 → 入库，逐阶段落盘可续跑。"""

from __future__ import annotations

import time
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.embed.embedder import get_embedder
from app.parsers.base import LayoutResult, parse_layout
from app.parsers.ocr import ocr_page_images
from app.parsers.structure import Section, build_section_tree
from app.splitter.chunker import Chunk, build_leaves, build_parent_blocks, embedding_text_for
from app.splitter.semantic import refine_leaves_semantically
from app.store import qdrant as qdrant_store
from app.store.registry import get_document, load_stage, save_stage, stage_done, update_document, update_task

logger = get_logger(__name__)


class PipelineError(Exception):
    pass


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
    try:
        # ---- 阶段 1：解析（版式层）----
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

        # ---- 阶段 2：章节树（结构层）----
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

        # ---- 阶段 3：切分（Small-to-Big + 语义精切）----
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

        # ---- 阶段 4：向量化（稠密 + 稀疏）----
        if task_id:
            _update_task_progress(task_id, "embed", 65, f"正在向量化 {len(chunks)} 个分块")
        update_document(doc_id, status="embedding")
        if not stage_done(doc_id, "vectors"):
            embedder = get_embedder()
            # 模型原生稠密 + 稀疏（text_type=document）；表格块用语义锚点增强文本嵌入
            # （锚点仅用于向量，展示/上下文仍用 chunk.content 原文）
            embed_texts = [embedding_text_for(c, doc_title=file_path.stem) for c in chunks]
            vectors, sparse_vectors = embedder.embed_texts_with_sparse(
                embed_texts, text_type="document"
            )
            save_stage(doc_id, "vectors", {"vectors": vectors, "sparse": sparse_vectors})
        else:
            stage = load_stage(doc_id, "vectors")
            vectors = stage["vectors"]
            sparse_vectors = stage.get("sparse")

        # 文档侧 IDF 重写（稀疏路治本）：入库前对稀疏 index 乘 IDF
        if sparse_vectors:
            from app.retrieval.idf import apply_idf

            sparse_vectors = [apply_idf(sv) for sv in sparse_vectors]

        if len(vectors) != len(chunks):
            raise PipelineError("向量数与分块数不一致，中止入库")
        _check_deadline(deadline, doc_id)

        # ---- 阶段 5：入库 ----
        if task_id:
            _update_task_progress(task_id, "index", 85, "正在写入向量库")
        doc = get_document(doc_id)
        qdrant_store.ensure_collection()
        n = qdrant_store.upsert_chunks(
            doc_id=doc_id,
            chunks=chunks,
            dense_vectors=vectors,
            doc_name=doc.filename if doc else file_path.name,
            org_id=org_id,
            visibility=visibility,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            sparse_vectors=sparse_vectors,
        )
        update_document(doc_id, status="indexed", chunk_count=n, error=None)
        if task_id:
            _update_task_progress(task_id, "index", 100, "入库完成")
            update_task(task_id, status="success", progress=100, message=f"解析完成，共 {n} 个分块")
        logger.info("doc %s indexed: %d chunks", doc_id, n)
        return {"doc_id": doc_id, "chunks": n}

    except Exception as e:
        logger.exception("ingest failed for doc %s", doc_id)
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
