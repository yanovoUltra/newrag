"""NDCG 网格搜索：扫描 RRF k（默认 30/60/100），用小型 golden 集评测检索质量。

用法（在 backend/ 目录下）：
    python scripts/eval_retrieval.py [--k 30,60,100] [--top-k 8] [--org default] [--visibility public]

- golden 集：scripts/golden/offline_257.json（257 题，预标注 relevant_chunk_ids）
- 相关集：直接使用 golden 预标注的 relevant_chunk_ids
- 指标：NDCG@top_k（rel=1，折扣 log2(i+1)），多查询取均值；同时输出命中率（Rec@top_k）
- 注意：检索走当前 .env 配置（rerank 按配置启用/关闭），会真实调用嵌入与 rerank API
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
from app.retrieval.search import hybrid_search  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402

GOLDEN_FILE = SCRIPT_DIR / "golden" / "offline_257.json"


def _norm(text: str) -> str:
    return (
        text.replace(",", "")
        .replace("，", "")
        .replace(" ", "")
        .replace("\u00a0", "")
        .replace("\n", "")
    )


def _load_corpus() -> dict[str, str]:
    """滚动全量 payload：chunk_id -> 归一化内容。"""
    client = qdrant_store.get_client()
    settings = get_settings()
    corpus: dict[str, str] = {}
    offset = None
    while True:
        res = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in res[0]:
            content = (p.payload or {}).get("content") or ""
            corpus[p.payload["chunk_id"]] = _norm(content)
        if res[1] is None:
            break
        offset = res[1]
    return corpus


def _ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """NDCG@k：rel=1，折扣 log2(i+1)；IDCG 为全相关时的理想分值。"""
    ranked = ranked[:k]
    dcg, idcg = 0.0, 0.0
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    for i in range(1, min(len(ranked), len(relevant)) + 1):
        idcg += 1.0 / math.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def main() -> int:
    parser = argparse.ArgumentParser(description="RRF k 网格搜索（NDCG）")
    parser.add_argument("--k", default="30,60,100", help="RRF k 候选，逗号分隔")
    parser.add_argument("--top-k", type=int, default=8, help="最终返回条数（NDCG 截止）")
    parser.add_argument("--org", default="default")
    parser.add_argument("--visibility", default="public")
    args = parser.parse_args()
    k_grid = [int(x) for x in args.k.split(",") if x.strip()]

    settings = get_settings()
    if not GOLDEN_FILE.exists():
        print(f"golden 文件不存在: {GOLDEN_FILE}")
        return 1
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))["questions"]

    print("golden 相关集已预标注（relevant_chunk_ids），跳过语料滚动。")

    # 预检：构建相关集，跳过相关集为空的条目
    embedder = get_embedder()
    items: list[tuple[str, set[str]]] = []
    for q in golden:
        rel = set(q.get("relevant_chunk_ids") or [])
        if not rel:
            print(f"  [跳过] 问题相关集为空: {q['question']}")
            continue
        items.append((q["question"], rel))
    if not items:
        print("无有效 golden 条目，退出")
        return 1
    print(f"有效 golden 条目: {len(items)}/{len(golden)}")

    print(f"网格: RRF k={k_grid}, NDCG@{args.top_k}, org={args.org}, visibility={args.visibility}\n")
    results: dict[int, list[float]] = {k: [] for k in k_grid}
    recall: dict[int, list[float]] = {k: [] for k in k_grid}
    for q in k_grid:
        settings.rrf_k = q
        for question, rel in items:
            dense, sparse = embedder.embed_texts_with_sparse([question], text_type="query")
            hits = hybrid_search(
                query_vector=dense[0],
                org_id=args.org,
                user_visibility=args.visibility,
                query_text=question,
                top_k=args.top_k,
                query_sparse=sparse[0] if sparse else None,
            )
            ranked = [h["chunk_id"] for h in hits]
            results[q].append(_ndcg_at_k(ranked, rel, args.top_k))
            recall[q].append(_recall_at_k(ranked, rel, args.top_k))

    print(f"{'RRF k':<8}{'NDCG@k':<12}{'Rec@k':<12}")
    print("-" * 32)
    best_k = max(k_grid, key=lambda k: sum(results[k]))
    for k in k_grid:
        n = sum(results[k]) / len(results[k])
        r = sum(recall[k]) / len(recall[k])
        marker = " *" if k == best_k else ""
        print(f"{k:<8}{n:<12.4f}{r:<12.4f}{marker}")
    print(f"\n最优 RRF k = {best_k}（按 NDCG 均值）")

    # 最优 k 下逐问题明细
    print(f"\n--- 最优 k={best_k} 逐问题明细 ---")
    settings.rrf_k = best_k
    for question, rel in items:
        dense, sparse = embedder.embed_texts_with_sparse([question], text_type="query")
        hits = hybrid_search(
            query_vector=dense[0],
            org_id=args.org,
            user_visibility=args.visibility,
            query_text=question,
            top_k=args.top_k,
            query_sparse=sparse[0] if sparse else None,
        )
        ranked = [h["chunk_id"] for h in hits]
        hit_docs = {}
        for h in hits[:args.top_k]:
            hit_docs[h["doc_name"]] = hit_docs.get(h["doc_name"], 0) + 1
        print(
            f"  NDCG={_ndcg_at_k(ranked, rel, args.top_k):.3f} 相关={len(rel)} "
            f"问题={question} 命中={dict(hit_docs)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
