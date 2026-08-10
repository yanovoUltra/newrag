"""方案对比评估：文档侧 IDF 重写的有无（NDCG@8 / Rec@8）。

数据状态：collection 稀疏向量已应用文档侧 IDF（生产状态）。
- 无 IDF 对照 = 查询稀疏除以 IDF（点乘还原原始得分，RRF 只用 rank，排序精确还原）。
- 结构块已收敛为单一降权行为（原 off/exclude/downgrade 多模式实测对召回无影响，已移除），
  本脚本按生产结构块设置（启用 + 降权）评估 IDF 单独的开/关差异。

Golden：scripts/golden/offline_257.json（257 题，预标注相关集，跳过 rel 为空项）
（茅台/比亚迪/宁德：营收 · 净利润 · 基本每股收益 · 加权平均净资产收益率）。
用法（backend/ 下）：python scripts/eval_idf_structural.py [--top-k 8]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.embed.embedder import get_embedder  # noqa: E402
from app.retrieval.rrf import rrf_fuse  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402
from app.store.registry import init_db  # noqa: E402
from eval_ablation import _norm, _load_corpus, _ndcg_at_k, _recall_at_k  # noqa: E402

GOLDEN_FILE = SCRIPT_DIR / "golden" / "offline_257.json"
_EXCLUDE = {"section"}


def _load_golden() -> list[dict]:
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))["questions"]


def _idf_map() -> dict[int, float]:
    """从 idf.json 读当前 index→idf 映射。"""
    p = get_settings().resolved_sparse_idf_path
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {int(k): float(v) for k, v in raw.items() if k != "_idf"}


def _divide_by_idf(sparse: dict, idf: dict[int, float]) -> dict:
    """查询稀疏除以 IDF（还原无文档侧 IDF 时的得分；RRF 只需排序）。"""
    return {
        "indices": list(sparse["indices"]),
        "values": [v / idf.get(int(i), 1.0) for i, v in zip(sparse["indices"], sparse["values"])],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="IDF × 结构块 方案对比")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--org", default="default")
    parser.add_argument("--visibility", default="public")
    args = parser.parse_args()

    settings = get_settings()
    init_db()
    golden = _load_golden()
    print(f"golden 相关集已预标注（relevant_chunk_ids），跳过语料滚动。")

    items: list[tuple[str, set[str]]] = []
    for q in golden:
        rel = set(q.get("relevant_chunk_ids") or [])
        if not rel:
            print(f"  [跳过 rel 为空] {q['question']}")
            continue
        items.append((q["question"], rel))
    print(f"有效 golden 条目: {len(items)}/{len(golden)}\n")

    idf = _idf_map()
    print(f"idf index 种类: {len(idf)}\n")

    embedder = get_embedder()
    per: dict[str, dict] = {}
    for q, _ in items:
        dense, sparse = embedder.embed_texts_with_sparse([q], text_type="query")
        d_vec, s_vec = dense[0], (sparse[0] if sparse else None)
        s_raw = _divide_by_idf(s_vec, idf) if s_vec else None
        per[q] = {"dense": d_vec, "sparse": s_vec, "sparse_raw": s_raw}
        print(f"  已嵌入: {q[:30]}")

    def _search():
        """按生产结构块设置（启用 + 降权）跑一趟，返回 {q: {d, s_idf, s_raw, t}}。"""
        res = {}
        for q, _ in items:
            d_vec = per[q]["dense"]
            s_vec = per[q]["sparse"]
            s_raw = per[q]["sparse_raw"]
            d = qdrant_store.search_dense(
                d_vec, org_id=args.org, user_visibility=args.visibility,
                top_k=settings.recall_dense_top_k, exclude_chunk_types=list(_EXCLUDE),
            )
            s_idf = qdrant_store.search_sparse(
                s_vec, org_id=args.org, user_visibility=args.visibility,
                top_k=settings.recall_sparse_top_k, exclude_chunk_types=list(_EXCLUDE),
            ) if s_vec else []
            s_raw_hits = qdrant_store.search_sparse(
                s_raw, org_id=args.org, user_visibility=args.visibility,
                top_k=settings.recall_sparse_top_k, exclude_chunk_types=list(_EXCLUDE),
            ) if s_raw else []
            t = qdrant_store.search_dense(
                d_vec, org_id=args.org, user_visibility=args.visibility,
                top_k=settings.recall_table_top_k, chunk_type="table",
                exclude_chunk_types=list(_EXCLUDE),
            )
            res[q] = {"d": [h["chunk_id"] for h in d],
                      "s_idf": [h["chunk_id"] for h in s_idf],
                      "s_raw": [h["chunk_id"] for h in s_raw_hits],
                      "t": [h["chunk_id"] for h in t]}
        return res

    R = _search()
    print("  已检索（生产结构块设置）")

    # 组装 IDF{on,off} 两套配置
    def _fuse(d, s, t, k=60, top_n=args.top_k):
        return [h["chunk_id"] for h in rrf_fuse([d, s, t], k=k, top_n=top_n)]

    configs = {}
    for idf_tag, s_key in (("IDF", "s_idf"), ("noIDF", "s_raw")):
        configs[idf_tag] = {}
        for q, _ in items:
            r = R[q]
            configs[idf_tag][q] = _fuse(
                [{"chunk_id": c} for c in r["d"]],
                [{"chunk_id": c} for c in r[s_key]],
                [{"chunk_id": c} for c in r["t"]],
            )
    # 生产默认 = IDF
    configs["生产默认(IDF)"] = configs["IDF"]

    print(f"\n{'配置':<16}{'NDCG@8':<12}{'Rec@8':<12}")
    print("-" * 40)
    order = ["noIDF", "IDF", "生产默认(IDF)"]
    results = {}
    for col in order:
        ranks = configs[col]
        ndcg = [_ndcg_at_k(ranks[q], rel, args.top_k) for q, rel in items]
        rec = [_recall_at_k(ranks[q], rel, args.top_k) for q, rel in items]
        results[col] = (sum(ndcg) / len(ndcg), sum(rec) / len(rec))
        print(f"{col:<16}{results[col][0]:<12.4f}{results[col][1]:<12.4f}")

    print("\n--- 逐问题 NDCG@8 ---")
    print(f"{'问题':<34}" + "".join(f"{c[:14]:>16}" for c in order))
    for q, rel in items:
        row = f"{q[:32]:<34}"
        for col in order:
            v = _ndcg_at_k(configs[col][q], rel, args.top_k)
            row += f"{v:>16.3f}"
        print(row)

    # 结论（IDF 净增益）
    print("\n=== IDF 净增益（NDCG@8） ===")
    base = results["noIDF"][0]
    print(f"净基线 noIDF: {base:.4f}")
    print(f"  IDF {results['IDF'][0]-base:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())