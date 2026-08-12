"""离线构建数字+单位短语倒排索引（number_phrase_index 表）。

全量扫描 Qdrant chunks，提取每个 chunk 的"数值+单位"短语（归一化后）→
写入 SQLite 倒排（phrase → chunk_id）。查询侧 hybrid_search 精确匹配后硬插候选。

- 章节父块（chunk_type=section）跳过（与叶子检索 _EXCLUDE_PARENT 口径一致）；
- 语料重建后需重跑本脚本（幂等：先清空全表再写入）。

用法（backend/ 下）：.\\\\.venv\\\\Scripts\\\\python.exe scripts\\\\build_number_phrase_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.fields.phrases import extract_number_phrases, norm_number_phrase  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402
from app.store.registry import init_db, replace_number_phrase_index  # noqa: E402


def main() -> int:
    init_db()
    settings = get_settings()
    client = qdrant_store.get_client()
    rows: list[tuple] = []
    offset = None
    n_chunks = 0
    n_skipped_section = 0
    while True:
        res = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=2000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in res[0]:
            pay = p.payload or {}
            cid = pay.get("chunk_id")
            content = pay.get("content")
            if not cid or content is None:
                continue
            if pay.get("chunk_type") == "section":
                n_skipped_section += 1
                continue
            n_chunks += 1
            seen: set[str] = set()
            for ph in extract_number_phrases(content):
                norm = norm_number_phrase(ph)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                rows.append((
                    norm,
                    cid,
                    pay.get("doc_id") or "",
                    pay.get("org_id") or "default",
                    pay.get("visibility") or "public",
                    pay.get("fiscal_year"),
                ))
        if res[1] is None:
            break
        offset = res[1]
    n = replace_number_phrase_index(rows)
    uniq = len({r[0] for r in rows})
    print(f"扫描 chunk: {n_chunks}（跳过 section 父块 {n_skipped_section}）")
    print(f"写入倒排行: {n}，唯一短语: {uniq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
