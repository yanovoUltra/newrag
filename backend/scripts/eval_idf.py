"""稀疏路噪声治理实验：IDF 修正（治本方案）。

背景（见 ablation_report.md 结论 4）：模型原生稀疏缺失 IDF 修正（Qdrant SparseVectorParams 无
modifier），泛化词（营业/收入/增长/同比/亿元）在稀疏里权重高、跨章节普遍出现 → 判别力低，
RRF 融合把稀疏路的无关块抬进 top_k，稀释 dense+sparse 结果（0.090 < dense-only 0.168）。

本脚本验证：给查询端稀疏权重叠加 IDF（index 级文档频率，指数项与 BM25 一致的平滑）后，
dense+sparse 是否由"稀释"变为"增益"，以及能否逼近生产完整（RRF+rerank）。

对比配置：
  dense-only / dense+sparse(原) / dense+sparse(IDF) / dense+sparse+table(IDF) / 生产完整
用法（backend/ 下）：python scripts/eval_idf.py [--top-k 8]
结果写入 eval_results 表（eval_name=ablation_idf）。
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
from app.retrieval.search import hybrid_search  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402
from app.store.registry import init_db, save_eval_result  # noqa: E402
from eval_ablation import _norm, _load_corpus, _ndcg_at_k, _recall_at_k  # noqa: E402

GOLDEN_FILE = SCRIPT_DIR / "golden" / "offline_257.json"
_EXCLUDE = {"section"}


def _collect_sparse_df(client, collection: str) -> tuple[int, dict[int, int]]:
    """滚动全 collection，统计每个稀疏 index 的文档频率 DF（每个 chunk 计 1 次）。"""
    df: dict[int, int] = {}
    n = 0
    offset = None
    while True:
        res = client.scroll(
            collection_name=collection, limit=1000, offset=offset,
            with_payload=False, with_vectors=True,
        )
        for p in res[0]:
            n += 1
            sv = p.vector.get("sparse") if isinstance(p.vector, dict) else None
            if sv is None:
                continue
            for idx in set(sv.indices):
                df[idx] = df.get(idx, 0) + 1
        if res[1] is None:
            break
        offset = res[1]
    return n, df


def _idf_map(n: int, df: dict[int, int]) -> dict[int, float]:
    return {idx: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for idx, f in df.items()}


def _apply_idf(sparse: dict, idf: dict[int, float], top_k: int | None = None) -> dict:
    """对查询稀疏权重叠加 IDF；可选按加权后分值做 top-k 截断（联合瘦身）。"""
    ind, val = sparse["indices"], sparse["values"]
    weighted = [(i, v * idf.get(i, 1.0)) for i, v in zip(ind, val)]
    weighted = [p for p in weighted if p[1] > 0]
    weighted.sort(key=lambda p: p[1], reverse=True)
    if top_k:
        weighted = weighted[:top_k]
    return {"indices": [p[0] for p in weighted], "values": [p[1] for p in weighted]}


def main() -> int:
    parser = argparse.ArgumentParser(description="稀疏路 IDF 修正实验")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--org", default="default")
    parser.add_argument("--visibility", default="public")
    parser.add_argument("--idf-topk", type=int, default=40, help="IDF 后查询稀疏 top-k 截断；0=不截断")
    args = parser.parse_args()

    settings = get_settings()
    init_db()
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))["questions"]

    items = []
    for q in golden:
        rel = set(q.get("relevant_chunk_ids") or [])
        if rel:
            items.append((q["question"], rel))
    print(f"有效 golden 条目: {len(items)}/{len(golden)}")

    # 统计 IDF
    client = qdrant_store.get_client()
    n, df = _collect_sparse_df(client, settings.qdrant_collection)
    idf = _idf_map(n, df)
    print(f"collection documents: {n}, 稀疏 index 种类: {len(df)}")

    embedder = get_embedder()
    per: dict[str, dict] = {}
    for question, _ in items:
        dense, sparse = embedder.embed_texts_with_sparse([question], text_type="query")
        d_vec, s_vec = dense[0], (sparse[0] if sparse else None)
        d_hits = qdrant_store.search_dense(
            query_vector=d_vec, org_id=args.org, user_visibility=args.visibility,
            top_k=settings.recall_dense_top_k, exclude_chunk_types=list(_EXCLUDE),
        )
        s_hits = (
            qdrant_store.search_sparse(
                query_sparse=s_vec, org_id=args.org, user_visibility=args.visibility,
                top_k=settings.recall_sparse_top_k, exclude_chunk_types=list(_EXCLUDE),
            )
            if s_vec else []
        )
        s_vec_idf = _apply_idf(s_vec, idf, top_k=args.idf_topk or None) if s_vec else None
        s_hits_idf = (
            qdrant_store.search_sparse(
                query_sparse=s_vec_idf, org_id=args.org, user_visibility=args.visibility,
                top_k=settings.recall_sparse_top_k, exclude_chunk_types=list(_EXCLUDE),
            )
            if s_vec_idf else []
        )
        t_hits = qdrant_store.search_dense(
            query_vector=d_vec, org_id=args.org, user_visibility=args.visibility,
            top_k=settings.recall_table_top_k, chunk_type="table", exclude_chunk_types=list(_EXCLUDE),
        )
        per[question] = {
            "dense": d_hits, "sparse": s_hits, "sparse_idf": s_hits_idf, "table": t_hits,
            "hybrid": hybrid_search(
                query_vector=d_vec, org_id=args.org, user_visibility=args.visibility,
                query_text=question, top_k=args.top_k, query_sparse=s_vec,
            ),
        }
        print(f"  已检索: {question[:30]}")

    def _rank(col_for_q):
        r = {}
        for q, _ in items:
            r[q] = [h["chunk_id"] for h in col_for_q(q)[: args.top_k]]
        return r

    ranked: dict[str, dict[str, list[str]]] = {
        "dense-only": _rank(lambda q: per[q]["dense"]),
        "dense+sparse(原)": _rank(lambda q: rrf_fuse([per[q]["dense"], per[q]["sparse"]], k=60, top_n=args.top_k)),
        "dense+sparse(IDF)": _rank(lambda q: rrf_fuse([per[q]["dense"], per[q]["sparse_idf"]], k=60, top_n=args.top_k)),
        "dense+sparse+table(IDF)": _rank(lambda q: rrf_fuse([per[q]["dense"], per[q]["sparse_idf"], per[q]["table"]], k=60, top_n=args.top_k)),
        "生产完整(rerank)": _rank(lambda q: per[q]["hybrid"]),
    }

    print(f"\n{'配置':<24}{'NDCG@8':<10}{'Rec@8':<10}")
    print("-" * 44)
    for col in ranked:
        ndcg = [_ndcg_at_k(ranked[col][q], rel, args.top_k) for q, rel in items]
        rec = [_recall_at_k(ranked[col][q], rel, args.top_k) for q, rel in items]
        print(f"{col:<24}{sum(ndcg)/len(ndcg):<10.4f}{sum(rec)/len(rec):<10.4f}")

    metrics = {
        "top_k": args.top_k, "idf_topk": args.idf_topk, "idf_smoothing": "BM25-like",
        "ndcg": {c: round(sum(_ndcg_at_k(ranked[c][q], rel, args.top_k) for q, rel in items) / len(items), 4) for c in ranked},
        "recall": {c: round(sum(_recall_at_k(ranked[c][q], rel, args.top_k) for q, rel in items) / len(items), 4) for c in ranked},
    }
    payload = {"corpus_docs": n, "sparse_index_types": len(df),
               "conclusion": "IDF 修正是否让 dense+sparse 由稀释变增益，见 ndcg 对比"}
    try:
        rec = save_eval_result("ablation_idf", f"org={args.org};top_k={args.top_k};idf_topk={args.idf_topk}", metrics, payload)
        print(f"\n已写入 eval_results: id={rec.id}")
    except Exception as e:
        print(f"写入失败: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())