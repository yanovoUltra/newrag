"""字段抽取回填：对已入库文档仅重跑字段抽取阶段（复用已保存的版式/章节中间产物，不重嵌入/重入库）。

背景：字段抽取（financial_fields 独立索引）在入库阶段 2.5 写入；已入库文档没有字段索引。
本脚本加载 data/pipeline/{doc_id}/ 下的 layout + sections 中间产物，抽取字段并写入 financial_fields 表。

用法（backend/ 下）：python scripts/backfill_fields.py [--doc <doc_id>] [--org default]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.logging import get_logger  # noqa: E402
from app.fields.extract import extract_fields  # noqa: E402
from app.pipelines.ingest import load_layout  # noqa: E402
from app.store.registry import (  # noqa: E402
    get_session,
    init_db,
    load_stage,
    replace_fields,
)

logger = get_logger(__name__)


def _indexed_docs(org_id: str | None = None) -> list:
    from sqlalchemy import select

    from app.models.entities import Document

    stmt = select(Document).where(Document.status == "indexed")
    if org_id:
        stmt = stmt.where(Document.org_id == org_id)
    with get_session() as s:
        return list(s.scalars(stmt).all())


def backfill(doc_id: str, org_id: str, visibility: str) -> int:
    layout = load_layout(doc_id)
    sections = None
    try:
        from app.parsers.structure import Section

        data = load_stage(doc_id, "sections") or {}
        sections = [Section.from_dict(s) for s in data.get("sections", [])]
    except Exception as e:
        logger.warning("doc %s 加载 sections 失败（无章节路径标注）: %s", doc_id, e)
    records = extract_fields(doc_id, layout, sections)
    n = replace_fields(doc_id, records, org_id, visibility)
    logger.info("doc %s 字段回填: %d", doc_id, n)
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description="字段抽取回填（复用版式/章节中间产物）")
    parser.add_argument("--doc", default=None, help="指定 doc_id；缺省回填全部 indexed 文档")
    parser.add_argument("--org", default=None, help="仅回填该 org 的文档")
    parser.add_argument("--visibility", default="public")
    args = parser.parse_args()

    init_db()

    if args.doc:
        n = backfill(args.doc, args.org or "default", args.visibility)
        print(f"doc {args.doc} 回填 {n} 条")
        return 0

    docs = _indexed_docs(args.org)
    total = 0
    for d in docs:
        total += backfill(d.id, d.org_id or args.org or "default", d.visibility or args.visibility)
    print(f"回填 {len(docs)} 个文档，共 {total} 条字段")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())