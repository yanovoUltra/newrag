"""离线入库管线：解析 → (OCR) → 章节树 → 切分 → 向量化 → 入库，逐阶段落盘可续跑。"""

from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.embed.embedder import get_embedder
from app.parsers.base import LayoutResult, parse_layout
from app.parsers.metadata import guess_fiscal_meta
from app.parsers.ocr import ocr_page_images
from app.parsers.structure import Section, build_section_tree
from app.splitter.chunker import Chunk, build_chunks
from app.store import qdrant as qdrant_store
from app.store.registry import load_stage, save_stage, stage_done, update_document, update_task

logger = get_logger(__name__)


class PipelineError(Exception):
    pass


def _update_task_progress(task_id: str, stage: str, progress: int, message: str = "") -> None:
    update_task(task_id, stage=stage, progress=progress, message=message)


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

        # ---- 阶段 2：章节树（结构层）----
        if task_id:
            _update_task_progress(task_id, "chunk", 40, "正在构建章节树")
        update_document(doc_id, status="chunking")
        if not stage_done(doc_id, "sections"):
            sections = build_section_tree(layout)
            save_stage(doc_id, "sections", {"sections": [s.to_dict() for s in sections]})
        else:
            sections = [Section.from_dict(s) for s in load_stage(doc_id, "sections")["sections"]]

        # ---- 阶段 3：切分（Small-to-Big）----
        if not stage_done(doc_id, "chunks"):
            chunks = build_chunks(doc_id, layout, sections, doc_title=file_path.stem)
            save_stage(doc_id, "chunks", {"chunks": [c.to_dict() for c in chunks]})
        else:
            chunks = [Chunk.from_dict(c) for c in load_stage(doc_id, "chunks")["chunks"]]

        if not chunks:
            raise PipelineError("切分结果为空（文档无可检索内容）")

        # ---- 阶段 4：向量化（稠密 + 稀疏）----
        if task_id:
            _update_task_progress(task_id, "embed", 65, f"正在向量化 {len(chunks)} 个分块")
        update_document(doc_id, status="embedding")
        if not stage_done(doc_id, "vectors"):
            embedder = get_embedder()
            # 模型原生稠密 + 稀疏（text_type=document）
            vectors, sparse_vectors = embedder.embed_texts_with_sparse(
                [c.content for c in chunks], text_type="document"
            )
            save_stage(doc_id, "vectors", {"vectors": vectors, "sparse": sparse_vectors})
        else:
            stage = load_stage(doc_id, "vectors")
            vectors = stage["vectors"]
            sparse_vectors = stage.get("sparse")

        if len(vectors) != len(chunks):
            raise PipelineError("向量数与分块数不一致，中止入库")

        # ---- 阶段 5：入库 ----
        if task_id:
            _update_task_progress(task_id, "index", 85, "正在写入向量库")
        doc = get_document_safe(doc_id)
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
        update_document(doc_id, status="failed", error=str(e))
        if task_id:
            update_task(task_id, status="failed", message=str(e))
        raise


def load_layout(doc_id: str) -> LayoutResult:
    data = load_stage(doc_id, "layout")
    if data is None:
        raise PipelineError("layout stage missing")
    return LayoutResult.from_dict(data)


def get_document_safe(doc_id: str):
    from app.store.registry import get_document

    return get_document(doc_id)
