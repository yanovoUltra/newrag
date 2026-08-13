"""批量入库已下载的财报（离线直接调 run_ingest，跳过 HTTP 上传）。

用法（backend/ 下）：python scripts/ingest_batch.py [--dir data/uploads]
- 对每个文件创建 registry 记录 + 任务，调 run_ingest 串行入库（各阶段落盘可续跑）。
- 打印每份的解析/切分/入库分块数；失败不中断，继续下一份。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.pipelines.ingest import run_ingest  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402
from app.store.registry import (  # noqa: E402
    create_document,
    create_task,
    init_db,
    list_documents,
)


def fiscal_year_for(name: str) -> int | None:
    """从文件名推断财年：年度报告→2025；招股书/美股 10-K→None。"""
    if "年度报告" in name or "年度报告摘要" in name:
        return 2025
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="批量入库财报")
    parser.add_argument("--dir", default=str(BACKEND_DIR.parent / "data" / "uploads"))
    args = parser.parse_args()
    init_db()

    src = Path(args.dir).resolve()
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".pdf", ".docx"))
    if not files:
        print(f"目录无可入库文件: {src}")
        return 1

    print(f"共 {len(files)} 个文件待入库")
    summary = []
    for f in files:
        content = f.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        fiscal_year = fiscal_year_for(f.name)
        # 同名幂等：sha256 命中且已 indexed 则跳过；半写状态则清向量重入
        doc = None
        for d in list_documents(org_id="default", limit=500):
            if d.sha256 == sha:
                doc = d
                break
        if doc is not None and doc.status == "indexed":
            print(f"[skip] {f.name} 已入库（{doc.chunk_count} chunks）")
            summary.append((f.name, doc.chunk_count))
            continue
        if doc is not None:
            qdrant_store.delete_doc(doc.id)  # 清掉半写向量，重新入库
        else:
            doc = create_document(
                filename=f.name,
                file_type=f.suffix.lstrip("."),
                org_id="default",
                visibility="public",
                size_bytes=len(content),
                sha256=sha,
                fiscal_year=fiscal_year,
            )
        task = create_task(doc.id, type_="ingest")
        try:
            t0 = time.time()
            res = run_ingest(
                doc.id, f, "default", "public", fiscal_year, None, task.id
            )
            n = res["chunks"]
            print(f"[OK] {f.name} -> {n} chunks in {time.time()-t0:.1f}s")
            summary.append((f.name, n))
        except Exception as e:
            print(f"[FAIL] {f.name} :: {type(e).__name__}: {str(e)[:200]}")
            summary.append((f.name, f"FAIL {str(e)[:80]}"))

    print("\n=== 汇总 ===")
    for name, n in summary:
        print(f"  {name}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
