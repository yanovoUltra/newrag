"""检索消融网格 + BM25 轻量基线对比（B1）。

对比各检索配置在小型 golden 集上的 NDCG@8 / Rec@8：
  1. dense-only         仅稠密路（排除 section/table），无 rerank
  2. sparse-only        仅模型原生稀疏路，无 rerank
  3. table-only         仅表格路，无 rerank
  4. dense+sparse       稠密+稀疏 RRF，无 rerank
  5. dense+sparse+table 三路 RRF，无 rerank
  6. 生产完整            三路 RRF + rerank（走 .env 配置的真实 rerank API）
  7. BM25 轻量基线      本地 BM25（k1=1.5,b=0.75，英文词+中文bigram），词法纯基线
  8. 加权配比            稠密/稀疏 RRF 分 convex 加权（w=0.3/0.5/0.7）

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
from app.retrieval.rrf import rrf_fuse  # noqa: E402
from app.retrieval.search import hybrid_search  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402
from app.store.registry import init_db, save_eval_result  # noqa: E402

GOLDEN_FILE = SCRIPT_DIR / "eval_golden.json"
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


# ---------------- BM25 轻量基线 ----------------

def _tokenize(text: str) -> list[str]:
    """轻量分词：英文/数字连续串 + 中文 2-gram（按连续中文字符滑窗）。"""
    tokens: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isascii() and ch.isalnum():
            j = i
            while j < n and text[j].isascii() and text[j].isalnum():
                j += 1
            tokens.append(text[i:j].lower())
            i = j
        elif "\u4e00" <= ch <= "\u9fff":
            j = i
            while j < n and "\u4e00" <= text[j] <= "\u9fff":
                j += 1
            seg = text[i:j]
            for t in range(len(seg) - 1):
                tokens.append(seg[t : t + 2])
            if len(seg) == 1:
                tokens.append(seg)
            i = j
        else:
            i += 1
    return tokens


class BM25:
    """标准 BM25（k1=1.5, b=0.75），就地实现，不做持久化。"""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len: list[int] = []
        self.docs: list[list[str]] = []
        df: dict[str, int] = {}
        for raw in corpus:
            toks = _tokenize(raw)
            self.docs.append(toks)
            self.doc_len.append(len(toks))
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.N = len(corpus)
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0
        self.idf = {t: math.log(1 + (self.N - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str) -> list[float]:
        q_tokens = _tokenize(query)
        scores = [0.0] * self.N
        for qt in set(q_tokens):
            idf = self.idf.get(qt, 0.0)
            if idf == 0.0:
                continue
            for i, doc in enumerate(self.docs):
                tf = doc.count(qt)
                if tf == 0:
                    continue
                dl = self.doc_len[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return scores


def _ranked_from_scores(ids: list[str], scores: list[float]) -> list[str]:
    return [cid for cid, _ in sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)]


def _weighted_fuse(dense: list[dict], sparse: list[dict], w: float, k: int = 60) -> list[dict]:
    """稠密/稀疏两路 RRF 分 convex 加权融合（非去重排序，权重配比实验用）。"""
    acc: dict[str, dict] = {}
    for route, weight in ((dense, w), (sparse, 1.0 - w)):
        for rank, hit in enumerate(route, start=1):
            cid = hit["chunk_id"]
            entry = acc.setdefault(cid, dict(hit))
            entry["fused_score"] = entry.get("fused_score", 0.0) + weight / (k + rank)
    return sorted(acc.values(), key=lambda h: h["fused_score"], reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="检索消融网格 + BM25 基线")
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
        rel = {cid for cid, info in corpus.items() if _norm(q["snippet"]) in info["content"]}
        if not rel:
            print(f"  [跳过] 相关集为空: {q['question']}")
            continue
        items.append((q["question"], rel))
    print(f"有效 golden 条目: {len(items)}/{len(golden)}\n")

    # BM25 构建（词法纯基线）
    bm25 = BM25([corpus[c]["raw"] for c in corpus_ids])
    settings.rrf_k = 60

    embedder = get_embedder()
    per_query: dict[str, dict[str, list[str]]] = {}
    for question, _ in items:
        dense, sparse = embedder.embed_texts_with_sparse([question], text_type="query")
        d_vec, s_vec = dense[0], (sparse[0] if sparse else None)

        dense_hits = qdrant_store.search_dense(
            query_vector=d_vec, org_id=args.org, user_visibility=args.visibility,
            top_k=settings.recall_dense_top_k, exclude_chunk_types=list(_EXCLUDE),
        )
        sparse_hits = (
            qdrant_store.search_sparse(
                query_sparse=s_vec, org_id=args.org, user_visibility=args.visibility,
                top_k=settings.recall_sparse_top_k, exclude_chunk_types=list(_EXCLUDE),
            )
            if s_vec
            else []
        )
        table_hits = qdrant_store.search_dense(
            query_vector=d_vec, org_id=args.org, user_visibility=args.visibility,
            top_k=settings.recall_table_top_k, chunk_type="table", exclude_chunk_types=list(_EXCLUDE),
        )
        bm_scores = bm25.score(question)
        bm_ranked = _ranked_from_scores(corpus_ids, bm_scores)[: settings.recall_sparse_top_k]

        per_query[question] = {
            "dense": [h["chunk_id"] for h in dense_hits],
            "sparse": [h["chunk_id"] for h in sparse_hits],
            "table": [h["chunk_id"] for h in table_hits],
            "dense_objs": dense_hits,
            "sparse_objs": sparse_hits,
            "table_objs": table_hits,
            "bm25_ids": bm_ranked,
            "hybrid": [h["chunk_id"] for h in hybrid_search(
                query_vector=d_vec, org_id=args.org, user_visibility=args.visibility,
                query_text=question, top_k=args.top_k, query_sparse=s_vec,
            )],
        }
        print(f"  已检索: {question[:30]}")

    def _eval(ranked: dict[str, list[str]]) -> dict[str, dict]:
        """返回 {col: {ndcg, rec, per_q}}。"""
        out: dict[str, dict] = {}
        for col, ranks in ranked.items():
            ndcg = [_ndcg_at_k(ranks[q], rel, args.top_k) for q, rel in items]
            rec = [_recall_at_k(ranks[q], rel, args.top_k) for q, rel in items]
            out[col] = {"ndcg": sum(ndcg) / len(ndcg), "rec": sum(rec) / len(rec), "per": ndcg}
        return out

    # 组装各列排序
    ranked: dict[str, dict[str, list[str]]] = {
        "dense-only": {},
        "sparse-only": {},
        "table-only": {},
        "dense+sparse": {},
        "dense+sparse+table": {},
        "生产完整(+rerank)": {},
        "BM25基线": {},
        "加权d0.3/s0.7": {},
        "加权d0.5/s0.5": {},
        "加权d0.7/s0.3": {},
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
        ranked["BM25基线"][q] = per_query[q]["bm25_ids"][: args.top_k]
        for w, col in ((0.3, "加权d0.3/s0.7"), (0.5, "加权d0.5/s0.5"), (0.7, "加权d0.7/s0.3")):
            fused = _weighted_fuse(per_query[q]["dense_objs"], per_query[q]["sparse_objs"], w)
            ranked[col][q] = [h["chunk_id"] for h in fused[: args.top_k]]

    results = _eval(ranked)

    print(f"\n{'配置':<22}{'NDCG@8':<12}{'Rec@8':<12}")
    print("-" * 46)
    ordered = [
        "dense-only", "sparse-only", "table-only", "dense+sparse", "dense+sparse+table",
        "生产完整(+rerank)", "BM25基线", "加权d0.3/s0.7", "加权d0.5/s0.5", "加权d0.7/s0.3",
    ]
    for col in ordered:
        r = results[col]
        print(f"{col:<22}{r['ndcg']:<12.4f}{r['rec']:<12.4f}")

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
    }
    payload = {
        "per_question_ndcg": {
            c: {q: round(v, 3) for (q, _), v in zip(items, results[c]["per"])} for c in ordered
        },
        "corpus_chunks": len(corpus_ids),
        "golden_questions": len(items),
        "bm25_params": {"k1": 1.5, "b": 0.75, "tokenizer": "ascii-word + zh-bigram"},
    }
    try:
        rec = save_eval_result("ablation", f"org={args.org};top_k={args.top_k}", metrics, payload)
        print(f"\n已写入 eval_results: id={rec.id}")
    except Exception as e:
        print(f"写入 eval_results 失败（忽略）: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
