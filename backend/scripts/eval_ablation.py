"""检索消融网格（B1）。

对比各检索配置在小型 golden 集上的 NDCG@8 / Rec@8 / Rec@1 / Prec@8 / MRR：
  1. dense-only         仅稠密路（排除 section/table），无 rerank
  2. sparse-only        仅模型原生稀疏路，无 rerank
  3. table-only         仅表格路，无 rerank
  4. dense+sparse       稠密+稀疏 RRF，无 rerank
  5. dense+sparse+table 三路 RRF，无 rerank
  6. 生产完整            三路 RRF + rerank（走 .env 配置的真实 rerank API）
注：BM25 词法基线与稠密/稀疏加权配比实验已删除——结论已固化（ablation_report.md）：
    BM25 无贡献、加权配比无意义，本语料（数字指标型+中文）不重做。

用法（backend/ 下）：
    python scripts/eval_ablation.py [--top-k 8] [--org default] [--visibility public]
结果同时写入 eval_results 表（eval_name=ablation）。
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
from app.pipelines.answer import _extract_year  # noqa: E402
from app.retrieval.rrf import rrf_fuse  # noqa: E402
from app.retrieval.search import hybrid_search  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402
from app.store.registry import init_db, save_eval_result  # noqa: E402

GOLDEN_FILE = SCRIPT_DIR / "golden" / "offline_257.json"
_EXCLUDE = {"section"}


def _norm(text: str) -> str:
    return (
        text.replace(",", "")
        .replace("，", "")
        .replace(" ", "")
        .replace("\u00a0", "")
        .replace("\n", "")
    )


def _load_corpus() -> dict[str, str]:
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
            payload = p.payload or {}
            content = payload.get("content") or ""
            corpus[payload["chunk_id"]] = {"content": _norm(content), "raw": content}
        if res[1] is None:
            break
        offset = res[1]
    return corpus


def _ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
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


def _precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & relevant) / k


def _mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="检索消融网格")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--org", default="default")
    parser.add_argument("--visibility", default="public")
    args = parser.parse_args()

    settings = get_settings()
    init_db()
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))["questions"]

    print("加载语料（滚动全量 payload）...")
    corpus = _load_corpus()
    corpus_ids = list(corpus.keys())
    print(f"语料 chunk 数: {len(corpus_ids)}")

    items: list[tuple[str, set[str]]] = []
    for q in golden:
        rel = set(q.get("relevant_chunk_ids") or [])
        if not rel:
            print(f"  [跳过] 相关集为空: {q['question']}")
            continue
        items.append((q["question"], rel))
    print(f"有效 golden 条目: {len(items)}/{len(golden)}\n")

    settings.rrf_k = 60

    embedder = get_embedder()
    per_query: dict[str, dict[str, list[str]]] = {}
    for question, _ in items:
        year = _extract_year(question)
        dense, sparse = embedder.embed_texts_with_sparse([question], text_type="query")
        d_vec, s_vec = dense[0], (sparse[0] if sparse else None)

        dense_hits = qdrant_store.search_dense(
            query_vector=d_vec, org_id=args.org, user_visibility=args.visibility,
            top_k=settings.recall_dense_top_k, exclude_chunk_types=list(_EXCLUDE),
            fiscal_year=year,
        )
        sparse_hits = (
            qdrant_store.search_sparse(
                query_sparse=s_vec, org_id=args.org, user_visibility=args.visibility,
                top_k=settings.recall_sparse_top_k, exclude_chunk_types=list(_EXCLUDE),
                fiscal_year=year,
            )
            if s_vec
            else []
        )
        table_hits = qdrant_store.search_dense(
            query_vector=d_vec, org_id=args.org, user_visibility=args.visibility,
            top_k=settings.recall_table_top_k, chunk_type="table", exclude_chunk_types=list(_EXCLUDE),
            fiscal_year=year,
        )

        # 生产完整列：hybrid_search（含 rerank + 字段 relay），并模拟与 answer.py 一致的
        # 年份回退——年份过滤后无非 relay 真实命中 → 回退全量再检索（修复 Rec@8 口径）
        hyb = hybrid_search(
            query_vector=d_vec, org_id=args.org, user_visibility=args.visibility,
            query_text=question, top_k=args.top_k, query_sparse=s_vec,
            fiscal_year=year,
        )
        if year and not any(not h.get("is_relay") for h in hyb):
            hyb = hybrid_search(
                query_vector=d_vec, org_id=args.org, user_visibility=args.visibility,
                query_text=question, top_k=args.top_k, query_sparse=s_vec,
                fiscal_year=None,
            )
        per_query[question] = {
            "dense": [h["chunk_id"] for h in dense_hits],
            "sparse": [h["chunk_id"] for h in sparse_hits],
            "table": [h["chunk_id"] for h in table_hits],
            "dense_objs": dense_hits,
            "sparse_objs": sparse_hits,
            "table_objs": table_hits,
            "hybrid": [h["chunk_id"] for h in hyb],
        }
        print(f"  已检索: {question[:30]}")

    def _eval(ranked: dict[str, list[str]]) -> dict[str, dict]:
        """返回 {col: {ndcg, rec, rec1, prec, mrr, per}}。"""
        out: dict[str, dict] = {}
        for col, ranks in ranked.items():
            ndcg = [_ndcg_at_k(ranks[q], rel, args.top_k) for q, rel in items]
            rec = [_recall_at_k(ranks[q], rel, args.top_k) for q, rel in items]
            rec1 = [_recall_at_k(ranks[q], rel, 1) for q, rel in items]
            prec = [_precision_at_k(ranks[q], rel, args.top_k) for q, rel in items]
            mrrs = [_mrr(ranks[q], rel) for q, rel in items]
            out[col] = {
                "ndcg": sum(ndcg) / len(ndcg),
                "rec": sum(rec) / len(rec),
                "rec1": sum(rec1) / len(rec1),
                "prec": sum(prec) / len(prec),
                "mrr": sum(mrrs) / len(mrrs),
                "per": ndcg,
            }
        return out

    # 组装各列排序
    ranked: dict[str, dict[str, list[str]]] = {
        "dense-only": {},
        "sparse-only": {},
        "table-only": {},
        "dense+sparse": {},
        "dense+sparse+table": {},
        "生产完整(+rerank)": {},
    }
    for q, rel in items:
        d_ids = per_query[q]["dense"]
        s_ids = per_query[q]["sparse"]
        t_ids = per_query[q]["table"]
        ranked["dense-only"][q] = d_ids[: args.top_k]
        ranked["sparse-only"][q] = s_ids[: args.top_k]
        ranked["table-only"][q] = t_ids[: args.top_k]
        ds = rrf_fuse([per_query[q]["dense_objs"], per_query[q]["sparse_objs"]], k=60, top_n=args.top_k)
        dst = rrf_fuse(
            [per_query[q]["dense_objs"], per_query[q]["sparse_objs"], per_query[q]["table_objs"]],
            k=60, top_n=args.top_k,
        )
        ranked["dense+sparse"][q] = [h["chunk_id"] for h in ds]
        ranked["dense+sparse+table"][q] = [h["chunk_id"] for h in dst]
        ranked["生产完整(+rerank)"][q] = per_query[q]["hybrid"]

    results = _eval(ranked)

    print(f"\n{'配置':<22}{'NDCG@8':<12}{'Rec@8':<12}{'Rec@1':<12}{'Prec@8':<12}{'MRR':<12}")
    print("-" * 70)
    ordered = [
        "dense-only", "sparse-only", "table-only", "dense+sparse", "dense+sparse+table",
        "生产完整(+rerank)",
    ]
    for col in ordered:
        r = results[col]
        print(
            f"{col:<22}{r['ndcg']:<12.4f}{r['rec']:<12.4f}{r['rec1']:<12.4f}"
            f"{r['prec']:<12.4f}{r['mrr']:<12.4f}"
        )

    print("\n--- 逐问题 NDCG@8 ---")
    print(f"{'问题':<34}" + "".join(f"{c[:10]:>12}" for c in ordered))
    for q, rel in items:
        row = f"{q[:32]:<34}"
        for col in ordered:
            row += f"{results[col]['per'][items.index((q, rel))]:>12.3f}"
        print(row)

    # 写入 eval_results 表
    metrics = {
        "top_k": args.top_k,
        "org": args.org,
        "ndcg": {c: round(results[c]["ndcg"], 4) for c in ordered},
        "recall": {c: round(results[c]["rec"], 4) for c in ordered},
        "recall1": {c: round(results[c]["rec1"], 4) for c in ordered},
        "precision": {c: round(results[c]["prec"], 4) for c in ordered},
        "mrr": {c: round(results[c]["mrr"], 4) for c in ordered},
    }
    payload = {
        "per_question_ndcg": {
            c: {q: round(v, 3) for (q, _), v in zip(items, results[c]["per"])} for c in ordered
        },
        "corpus_chunks": len(corpus_ids),
        "golden_questions": len(items),
    }
    try:
        rec = save_eval_result("ablation", f"org={args.org};top_k={args.top_k}", metrics, payload)
        print(f"\n已写入 eval_results: id={rec.id}")
    except Exception as e:
        print(f"写入 eval_results 失败（忽略）: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
