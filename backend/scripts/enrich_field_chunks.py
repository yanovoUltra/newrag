"""全量回填所有文档字段的 source_chunk_id（存量修复/补跑）。

核心定位逻辑已下沉到 app/fields/backfill.py（ingest 主流程阶段 3.5 也调用它，
新上传文档不再依赖本脚本）；本脚本仅作"历史文档全量补跑"入口。

用法（backend/ 下）：python scripts/enrich_field_chunks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import select  # noqa: E402

from app.fields.backfill import backfill_doc_fields  # noqa: E402
from app.models.entities import FinancialField  # noqa: E402
from app.store.registry import get_session, init_db, load_stage  # noqa: E402


def main() -> int:
    init_db()
    with get_session() as s:
        doc_ids = list(s.scalars(select(FinancialField.doc_id).distinct()).all())

    total_updated = 0
    for doc_id in doc_ids:
        data = load_stage(doc_id, "chunks")
        if not data or not data.get("chunks"):
            print(f"doc {doc_id[:8]} 无 chunks 阶段产物，跳过")
            continue
        from app.splitter.chunker import Chunk

        chunks = [Chunk.from_dict(c) for c in data["chunks"]]
        n = backfill_doc_fields(doc_id, chunks)
        total_updated += n
        print(f"doc {doc_id[:8]} 回填 {n} 条")
    print(f"回填 source_chunk_id: {total_updated} 条（本次新增）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
