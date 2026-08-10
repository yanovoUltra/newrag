"""重嵌入：为已有文档的表格块补“语义锚点”向量（公司/章节→表格），不重解析、不重建 payload。

背景：表块语义锚点（embedding_text_for）只改变**嵌入文本**，展示/上下文仍用 chunk.content 原文。
对已入库文档，需用增强文本重新生成 dense/sparse 向量并覆盖原向量点（chunk_id 不变→点 id 幂等）。

用法（backend/ 下）：python scripts/reembed_tables.py [--doc <doc_id>] [--org default]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import select  # noqa: E402

from app.core.logging import get_logger  # noqa: E402
from app.embed.embedder import get_embedder  # noqa: E402
from app.models.entities import Document  # noqa: E402
from app.splitter.chunker import Chunk, embedding_text_for  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402
from app.store.registry import get_session, init_db, load_stage  # noqa: E402

logger = get_logger(__name__)


def _indexed_docs(org_id: str | None = None) -> list[Document]:
    stmt = select(Document).where(Document.status == "indexed")
    if org_id:
        stmt = stmt.where(Document.org_id == org_id)
    with get_session() as s:
        return list(s.scalars(stmt).all())


def reembed(doc: Document, org_id: str, visibility: str, fiscal_year: int | None) -> int:
    data = load_stage(doc.id, "chunks")
    if not data or not data.get("chunks"):
        logger.warning("doc %s 缺少 chunks 阶段产物，跳过", doc.id)
        return 0
    chunks = [Chunk.from_dict(c) for c in data["chunks"]]
    # 仅表块需要重锚定：embedding_text_for 对非表块返回 content 原文（与入库时一致），
    # 其向量已正确，无需重嵌；只重嵌表块并幂等覆盖其点，非表块点保持不动。
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    if not table_chunks:
        logger.info("doc %s 无表块，跳过", doc.id)
        return 0
    doc_title = Path(doc.filename).stem
    embed_texts = [embedding_text_for(c, doc_title=doc_title) for c in table_chunks]

    embedder = get_embedder()
    vectors, sparse_vectors = embedder.embed_texts_with_sparse(embed_texts, text_type="document")
    if sparse_vectors:
        from app.retrieval.idf import apply_idf

        sparse_vectors = [apply_idf(sv) for sv in sparse_vectors]

    qdrant_store.ensure_collection()
    n = qdrant_store.upsert_chunks(
        doc_id=doc.id,
        chunks=table_chunks,
        dense_vectors=vectors,
        doc_name=doc.filename,
        org_id=org_id,
        visibility=visibility,
        fiscal_year=fiscal_year,
        fiscal_quarter=getattr(doc, "fiscal_quarter", None),
        sparse_vectors=sparse_vectors,
    )
    logger.info("doc %s 重嵌入完成: %d 个表块（已加语义锚点）", doc.id, n)
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description="重嵌入表格语义锚点向量")
    parser.add_argument("--doc", default=None, help="指定 doc_id；缺省重嵌入全部 indexed 文档")
    parser.add_argument("--org", default=None)
    args = parser.parse_args()

    init_db()
    qdrant_store.ensure_collection()

    if args.doc:
        with get_session() as s:
            doc = s.get(Document, args.doc)
        if not doc:
            print(f"文档不存在: {args.doc}")
            return 1
        n = reembed(doc, doc.org_id or "default", doc.visibility or "public", doc.fiscal_year)
        print(f"doc {args.doc} 重嵌入 {n} 块")
        return 0

    docs = _indexed_docs(args.org)
    total = 0
    ok = 0
    for d in docs:
        try:
            total += reembed(d, d.org_id or args.org or "default", d.visibility or "public", d.fiscal_year)
            ok += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("doc %s 重嵌入失败（跳过，可重跑续传）: %s", d.id, e)
    print(f"重嵌入 {ok}/{len(docs)} 个文档成功，共 {total} 块（失败文档可重跑续传）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())