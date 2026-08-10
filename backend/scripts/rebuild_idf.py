"""离线重建文档侧 IDF（稀疏路治本）。

用法（backend/ 下）：
    python scripts/rebuild_idf.py --rewrite
      - 统计当前 collection 稀疏 index 的文档频率 → 计算 IDF → 写 data/idf.json；
      - --rewrite 时同时重写已入库点的稀疏权重（应用 IDF，保留稠密与 payload）。

注意：--rewrite 会改动 collection 的 sparse 向量，建议在验证 IDF 效果后再批量执行；
之后的新入库会自动应用同一 IDF（idf.json 存在即生效）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.retrieval.idf import build_idf  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="重建文档侧 IDF")
    parser.add_argument("--rewrite", action="store_true", help="同时重写已入库点的稀疏权重")
    args = parser.parse_args()
    n = build_idf(rewrite=args.rewrite)
    print(f"IDF 重建完成：{n} 个 index 种类" + ("，已重写 collection 稀疏权重" if args.rewrite else "（未重写，用 --rewrite 生效）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())