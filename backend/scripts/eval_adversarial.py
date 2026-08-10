"""对抗性/压力测试评测：4 类边界用例 × 4 指标 + 特殊检验，落库 eval_results。

用法（backend/ 下）：
    python scripts/eval_adversarial.py [--top-k 10] [--k 5,8,10] [--org default] [--visibility public] [--hyde]

评测流程：
- 读取 scripts/eval_adversarial_golden.json（含 relevant_chunk_ids + 类别 + special 标记）
- 每题：提取显式年份 → hybrid_search（带 fiscal_year 过滤）→ Top-K
- --hyde：对 D 类（跨页长文本/综述型）用例先用 LLM 生成假设文档（HyDE）再检索，
  验证综述型问题（"总结 MD&A 中关于风险的看法"）的召回修复
- 指标：Recall@K / Precision@K / NDCG@K / MRR（按类别分组汇总 + 全集）
- 特殊检验（对应被击穿漏洞）：
  top3           数值精度：相关块是否进入 Top-3（精确数字表不能被语义淹没）
  no_iphone_only 否定词：iPhone 独占块（含 iPhone 不含 Mac/Net sales）是否混入 Top-3
  year_filter    时间混淆：Top-K 中 fiscal_year 非目标年份的块占比（混杂率，应 0）
  跨页类别        D 类：相关块召回比例（Recall@K）——多块全召回而非只抓第一段
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.embed.embedder import get_embedder  # noqa: E402
from app.pipelines.answer import _extract_year  # noqa: E402
from app.retrieval.router import generate_hypothetical_document  # noqa: E402
from app.retrieval.search import hybrid_search  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402
from app.store.registry import init_db, save_eval_result  # noqa: E402

GOLDEN_FILE = SCRIPT_DIR / "golden" / "adversarial_v2.json"

CAT_LABEL = {
    "numeric_precision": "A 数值精度",
    "negation_trap": "B 否定词",
    "year_confusion": "C 时间混淆",
    "long_text_crosspage": "D 跨页长文本",
}


def _precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / k if k > 0 else 0.0


def _recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def _ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    ranked = ranked[:k]
    dcg, idcg = 0.0, 0.0
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    for i in range(1, min(len(ranked), len(relevant)) + 1):
        idcg += 1.0 / math.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0


def _mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def _payload_years(hits: list[dict]) -> dict[str, int | None]:
    """批量拉取 top 命中块的 fiscal_year（用于年份混杂率检验）。"""
    by_doc: dict[str, list[str]] = {}
    for h in hits:
        by_doc.setdefault(h["doc_id"], []).append(h["chunk_id"])
    payloads: dict[str, dict] = {}
    for doc_id, cids in by_doc.items():
        payloads.update(qdrant_store.fetch_payloads(doc_id, cids))
    return {h["chunk_id"]: payloads.get(h["chunk_id"], {}).get("fiscal_year") for h in hits}


def main() -> int:
    parser = argparse.ArgumentParser(description="对抗性/压力测试评测")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k", default="5,8,10")
    parser.add_argument("--org", default="default")
    parser.add_argument("--visibility", default="public")
    parser.add_argument("--hyde", action="store_true", help="D 类综述型用例用 HyDE 假设文档检索")
    args = parser.parse_args()
    k_list = [int(x) for x in args.k.split(",") if x.strip()]
    k_list = [k for k in k_list if 0 < k <= args.top_k]
    if not k_list:
        print("无有效 K 值，退出")
        return 1

    init_db()
    if not GOLDEN_FILE.exists():
        print(f"golden 文件不存在，请先运行 gen_adversarial_golden.py: {GOLDEN_FILE}")
        return 1
    data = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    golden = data["questions"]
    print(f"加载对抗性 golden {len(golden)} 题")

    embedder = get_embedder()
    agg_all = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_list}
    agg_cat: dict[str, dict] = {}
    agg_mrr: list[float] = []
    special: dict[str, dict] = {}
    details: list[dict] = []

    for q in golden:
        cid_ = q["id"]
        category = q["category"]
        question = q["question"]
        relevant = set(q.get("relevant_chunk_ids") or [])
        year = _extract_year(question)
        # HyDE：D 类（综述型/跨页长文本）先由 LLM 生成假设文档，再用其检索
        hyde_doc = None
        if args.hyde and category == "long_text_crosspage":
            hyde_doc = asyncio.run(generate_hypothetical_document(question))
        dense, sparse = embedder.embed_texts_with_sparse([hyde_doc or question], text_type="query")
        hits = hybrid_search(
            query_vector=dense[0],
            org_id=args.org,
            user_visibility=args.visibility,
            query_text=question,
            top_k=args.top_k,
            query_sparse=sparse[0] if sparse else None,
            fiscal_year=year,  # 时间混淆检验：显式年份 → 过滤
        )
        ranked = [h["chunk_id"] for h in hits]
        mrr = _mrr(ranked, relevant)
        agg_mrr.append(mrr)
        row = {
            "id": cid_,
            "category": category,
            "question": question,
            "year_filtered": year,
            "hyde_used": hyde_doc is not None,
            "n_rel": len(relevant),
            "mrr": mrr,
            "hit": sorted(set(ranked) & relevant)[:5],
            "top3_hit": bool(set(ranked[:3]) & relevant),
        }
        # 特殊检验
        special[cid_] = {}
        if q.get("special") == "top3":
            special[cid_]["top3_hit"] = row["top3_hit"]
        if q.get("special") == "no_iphone_only":
            top3 = hits[:3]
            iphone_only = [
                h for h in top3
                if "iPhone" in (h.get("content") or "")
                and "Mac" not in (h.get("content") or "")
                and "Net sales" not in (h.get("content") or "")
            ]
            special[cid_]["iphone_only_in_top3"] = len(iphone_only)
        if q.get("special") == "year_filter" and year:
            yrs = _payload_years(hits[: args.top_k])
            mix = sum(1 for v in yrs.values() if v != year) / max(len(yrs), 1)
            special[cid_]["year_mix_rate"] = round(mix, 4)
        row["per_k"] = {
            str(k): {
                "precision": _precision_at_k(ranked, relevant, k),
                "recall": _recall_at_k(ranked, relevant, k),
                "ndcg": round(_ndcg_at_k(ranked, relevant, k), 4),
            }
            for k in k_list
        }
        details.append(row)
        agg_cat.setdefault(category, {k: {"precision": [], "recall": [], "ndcg": []} for k in k_list})
        agg_cat[category].setdefault("_mrr", [])
        agg_cat[category]["_mrr"].append(mrr)
        for k in k_list:
            agg_all[k]["precision"].append(row["per_k"][str(k)]["precision"])
            agg_all[k]["recall"].append(row["per_k"][str(k)]["recall"])
            agg_all[k]["ndcg"].append(row["per_k"][str(k)]["ndcg"])
            agg_cat[category][k]["precision"].append(row["per_k"][str(k)]["precision"])
            agg_cat[category][k]["recall"].append(row["per_k"][str(k)]["recall"])
            agg_cat[category][k]["ndcg"].append(row["per_k"][str(k)]["ndcg"])

    n = len(details)
    if n == 0:
        print("无有效评测条目，退出")
        return 1

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    k8 = 8 if 8 in k_list else k_list[-1]
    print(f"\n===== 对抗性/压力测试评测（{n} 题）{'[HyDE 开启: D 类综述型]' if args.hyde else '[baseline]'} =====")
    metrics: dict = {"eval_name": "adversarial", "num_questions": n, "top_k": args.top_k}

    # 按类别汇总
    print(f"\n{'类别':<12}{'题数':<6}{'Precision@8':<12}{'Recall@8':<12}{'NDCG@8':<12}{'MRR':<10}")
    print("-" * 64)
    for cat in ["numeric_precision", "negation_trap", "year_confusion", "long_text_crosspage"]:
        if cat not in agg_cat:
            continue
        rows = agg_cat[cat]
        p = _mean(rows[k8]["precision"])
        r = _mean(rows[k8]["recall"])
        nd = _mean(rows[k8]["ndcg"])
        mr = _mean(rows["_mrr"])
        cnt = len(rows[k8]["precision"])
        metrics[f"{cat}_precision_at_8"] = round(p, 4)
        metrics[f"{cat}_recall_at_8"] = round(r, 4)
        metrics[f"{cat}_ndcg_at_8"] = round(nd, 4)
        metrics[f"{cat}_mrr"] = round(mr, 4)
        print(f"{CAT_LABEL.get(cat, cat):<12}{cnt:<6}{p:<12.4f}{r:<12.4f}{nd:<12.4f}{mr:<10.4f}")

    print(f"\n{'K':<6}{'Precision@K':<14}{'Recall@K':<12}{'NDCG@K':<12}")
    print("-" * 44)
    for k in k_list:
        p = _mean(agg_all[k]["precision"])
        r = _mean(agg_all[k]["recall"])
        nd = _mean(agg_all[k]["ndcg"])
        metrics[f"precision_at_{k}"] = round(p, 4)
        metrics[f"recall_at_{k}"] = round(r, 4)
        metrics[f"ndcg_at_{k}"] = round(nd, 4)
        print(f"{k:<6}{p:<14.4f}{r:<12.4f}{nd:<12.4f}")
    mrr = _mean(agg_mrr)
    metrics["mrr"] = round(mrr, 4)
    print(f"\nMRR = {mrr:.4f}")

    # 特殊检验汇总
    print("\n===== 特殊检验（漏洞验证）=====")
    top3_ok = [s for cid_, s in special.items() if "top3_hit" in s]
    if top3_ok:
        rate = sum(1 for s in top3_ok if s["top3_hit"]) / len(top3_ok)
        metrics["special_top3_hit_rate"] = round(rate, 4)
        print(f"[数值精度] 相关块进入 Top-3 比例: {rate:.2%}（{len(top3_ok)} 题）")
        for cid_, s in special.items():
            if "top3_hit" in s:
                print(f"    {cid_}: Top3命中={s['top3_hit']}")
    iphone_checks = [s for cid_, s in special.items() if "iphone_only_in_top3" in s]
    if iphone_checks:
        total = sum(s["iphone_only_in_top3"] for s in iphone_checks)
        metrics["special_iphone_only_in_top3"] = total
        print(f"[否定词] '除了iPhone' Top-3 中 iPhone 独占块数（应 0）: {total}")
    year_checks = [s for cid_, s in special.items() if "year_mix_rate" in s]
    if year_checks:
        for cid_, s in special.items():
            if "year_mix_rate" in s:
                metrics[f"special_year_mix_{cid_}"] = s["year_mix_rate"]
                print(f"[时间混淆] {cid_} 非目标年份块占比（应 0）: {s['year_mix_rate']:.4f}")
    cat_d = agg_cat.get("long_text_crosspage")
    if cat_d:
        r8 = _mean(cat_d[k8]["recall"])
        metrics["special_crosspage_recall_at_8"] = round(r8, 4)
        print(f"[跨页长文本] D 类 Recall@8（分散多块召回比例）: {r8:.4f}")
        for d in details:
            if d["category"] == "long_text_crosspage":
                hy = " HyDE" if d.get("hyde_used") else ""
                print(f"    {d['id']}{hy}: n_rel={d['n_rel']} recall@8={d['per_k']['8']['recall']:.3f} mrr={d['mrr']:.3f}")

    rec = save_eval_result(
        eval_name="adversarial",
        scope=f"adversarial golden={n} questions, org={args.org}, vis={args.visibility}, top_k={args.top_k}",
        metrics=metrics,
        payload={"details": details, "k_list": k_list, "special": special},
    )
    print(f"\n已落库 eval_results id={rec.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
