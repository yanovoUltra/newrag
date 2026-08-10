"""给已入库点补 is_structural 标记（无需重入库）。

现有 collection 的点在结构块检测上线前已写入，缺少 is_structural 字段。本脚本基于 content
调用 is_structural_chunk 判定，用 Qdrant set_payload 批量补标（保留原向量与其余 payload）。

用法（backend/ 下）：python scripts/migrate_structural.py
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.parsers.structural import is_structural_chunk  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402


def main() -> int:
    settings = get_settings()
    client = qdrant_store.get_client()
    marked = 0
    total = 0
    offset = None
    while True:
        res = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1000, offset=offset, with_payload=True, with_vectors=False,
        )
        batch = []
        for p in res[0]:
            total += 1
            content = (p.payload or {}).get("content") or ""
            flag = bool(is_structural_chunk(content))
            if flag:
                marked += 1
            batch.append({"id": p.id, "is_structural": flag})
        for i in range(0, len(batch), 256):
            chunk = batch[i : i + 256]
            true_ids = [b["id"] for b in chunk if b["is_structural"]]
            false_ids = [b["id"] for b in chunk if not b["is_structural"]]
            if true_ids:
                client.set_payload(
                    collection_name=settings.qdrant_collection,
                    payload={"is_structural": True},
                    points=true_ids,
                )
            if false_ids:
                client.set_payload(
                    collection_name=settings.qdrant_collection,
                    payload={"is_structural": False},
                    points=false_ids,
                )
        if res[1] is None:
            break
        offset = res[1]
    print(f"补标完成：{total} 点，其中结构块 {marked} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())