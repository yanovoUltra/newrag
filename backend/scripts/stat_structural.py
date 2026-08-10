"""统计当前 collection 的结构块分布，供人工核对结构块是否为纯版式块。

输出：结构块总数、按 chunk_type 分布、结构块内容片段（人工核对是否纯版式块）。

用法（backend/ 下）：python scripts/stat_structural.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402


def main() -> int:
    settings = get_settings()
    client = qdrant_store.get_client()
    total = 0
    marked = 0
    by_type: Counter = Counter()
    samples = []
    offset = None
    while True:
        res = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1000, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in res[0]:
            total += 1
            pl = p.payload or {}
            if pl.get("is_structural"):
                marked += 1
                by_type[pl.get("chunk_type", "text")] += 1
                if len(samples) < 50:
                    content = (pl.get("content") or "").replace("\n", " ").strip()
                    samples.append(
                        f"[{pl.get('doc_name')} p{pl.get('page')} {pl.get('chunk_type')}] "
                        f"tok={pl.get('token_count')} :: {content[:120]}"
                    )
        if res[1] is None:
            break
        offset = res[1]

    print(f"总点数：{total}")
    print(f"结构块总数：{marked}（占比 {marked / max(total, 1):.2%}）")
    print("按 chunk_type 分布：", dict(by_type) if by_type else "（无）")
    print("\n结构块内容片段（前 50 条）：")
    for s in samples:
        print("  " + s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())