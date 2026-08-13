"""指标题/非指标题分组评测（生产真实路径口径）。

输出两组指标：
- 非指标题：NDCG@8 / MRR / Prec@8（纯语义检索水平）
- 指标题：NDCG@8 / MRR / Prec@8 + 三个抽取链路准确率：
    · 科目同义匹配准确率：extract_metric_from_question 提取科目与 golden 规范 key 一致
    · 时间周期准确率：提取年份与 golden.year 一致（golden.year 非空时）
    · 财务数值提取准确率：字段索引（科目+年份+主体）取回数值与 golden.snippet 数值匹配

检索走生产真实路径（_search_plan 保守近似，与 eval_ablation hyb_prod 同口径），
保证 NDCG/MRR/Prec 与简历报告（非指标 0.4994/MRR 0.8239）可比。

用法（backend/ 下）：
    python scripts/eval_metrics.py [--golden scripts/golden/eval_set.json] [--top-k 8]
    [--concurrency 4] [--summary]   # --summary 开启摘要锚点（默认关）
结果写入 eval_results 表（eval_name=metrics）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import golden_utils  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.fields.metrics import extract_metric_from_question  # noqa: E402
from app.fields.subject import find_subject_company  # noqa: E402
from app.pipelines.answer import _extract_year, _search_plan  # noqa: E402
from app.retrieval.router import _is_summary_query, classify_query_type_keyword  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402
from app.store.registry import (  # noqa: E402
    init_db,
    list_field_companies,
    query_fields,
    save_eval_result,
)

GOLDEN_FILE = SCRIPT_DIR / "golden" / "eval_set.json"


def _norm_num(s: str) -> float | None:
    try:
        return float(
            str(s).replace(",", "").replace("，", "").replace("%", "").strip()
        )
    except (ValueError, TypeError):
        return None


def _num_eq(value, snippet: str | None) -> bool:
    """字段数值与 golden snippet 数值匹配（相对误差 < 1e-6）。"""
    if value is None or not snippet:
        return False
    s = _norm_num(snippet)
    if s is None:
        return False
    if value == s:
        return True
    return abs(value - s) / max(abs(s), 1e-9) < 1e-6


def _ndcg(ranked: list[str], rel: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, c in enumerate(ranked[:k], 1) if c in rel)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(rel)) + 1))
    return dcg / idcg if idcg else 0.0


def _mrr(ranked: list[str], rel: set[str]) -> float:
    for i, c in enumerate(ranked, 1):
        if c in rel:
            return 1.0 / i
    return 0.0


def _prec(ranked: list[str], rel: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & rel) / k


def _recall(ranked: list[str], rel: set[str], k: int) -> float:
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


def _all_hit(ranked: list[str], rel: set[str], k: int) -> float:
    """严格覆盖：全部相关块都进入 top-k；相关集大于 k 时按定义不可达并记 0。"""
    return float(bool(rel) and rel.issubset(set(ranked[:k])))


def _expand_rel_with_parents(rel: set[str], corpus: dict) -> set[str]:
    out = set(rel)
    for cid in rel:
        pid = corpus.get(cid, {}).get("parent_id")
        if pid:
            out.add(pid)
    return out


async def _prod_real(
    question: str, org_id: str, user_visibility: str, top_k: int, year: int | None,
    *, full_router: bool = False, rerank_candidates: int | None = None,
) -> list[dict]:
    """生产检索路径；默认使用可重现近似，full_router 启用真实 LLM 路由。"""
    from app.retrieval.router import RoutePlan, route_query

    if full_router:
        plan = await route_query(question)
    else:
        qtype = classify_query_type_keyword(question)
        plan = RoutePlan(
            intent="factual",
            complexity="simple",
            query_type=qtype,
            needs_hyde=_is_summary_query(question),
        )
    return await _search_plan(
        plan, question, org_id, user_visibility, top_k, fiscal_year=year,
        rerank_candidates=rerank_candidates,
    )


def _metric_extraction(q: dict, org_id: str, vis: str) -> dict:
    """指标题三个抽取链路指标：科目同义 / 时间周期 / 财务数值。"""
    key, year = extract_metric_from_question(q.get("question") or "")
    subject_ok = bool(key) and key == q.get("metric")
    year_ok = True
    if q.get("year") is not None:
        year_ok = year == q.get("year")
    value_ok = False
    if key:
        comp = find_subject_company(q.get("question") or "", list_field_companies(org_id, vis))
        rows = query_fields([key], org_id, vis, year=year or q.get("year"), company=comp, limit=1)
        value_ok = _num_eq(rows[0].value if rows else None, q.get("snippet"))
    return {"subject_ok": subject_ok, "year_ok": year_ok, "value_ok": value_ok}


async def main() -> int:
    parser = argparse.ArgumentParser(description="指标/非指标分组评测")
    parser.add_argument("--golden", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--org", default="default")
    parser.add_argument("--visibility", default="public")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--summary", action="store_true", help="开启摘要锚点（默认关，基线口径）")
    parser.add_argument("--only-non-metric", action="store_true", help="只跑非指标题（诊断用）")
    parser.add_argument("--only-metric", action="store_true", help="只跑指标题（诊断用）")
    parser.add_argument("--full-router", action="store_true", help="调用真实 LLM 意图路由（生产口径，较慢）")
    parser.add_argument("--rerank-candidates", type=int, default=None,
                        help="精排候选池大小（默认使用生产配置）")
    parser.add_argument("--dump", default=None, help="输出逐题 NDCG/MRR 到 json（诊断用）")
    args = parser.parse_args()
    settings = get_settings()
    settings.summary_enabled = args.summary
    init_db()
    golden = json.loads((Path(args.golden) if args.golden else GOLDEN_FILE).read_text(encoding="utf-8"))["questions"]
    print(f"golden: {len(golden)} 题（指标 {sum(1 for q in golden if q.get('type','metric')=='metric')} / "
          f"非指标 {sum(1 for q in golden if q.get('type','metric')!='metric')}）")

    fp_index = golden_utils.FingerprintIndex.build(args.org, args.visibility)
    corpus: dict[str, dict] = {}
    client = qdrant_store.get_client()
    offset = None
    while True:
        res = client.scroll(
            collection_name=settings.qdrant_collection, limit=1000, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in res[0]:
            pay = p.payload or {}
            corpus[pay["chunk_id"]] = {"doc_id": pay.get("doc_id"), "parent_id": pay.get("parent_id")}
        if res[1] is None:
            break
        offset = res[1]

    items: list[dict] = []
    for q in golden:
        rel = golden_utils.resolve_rel(q, fp_index)
        if q.get("doc_id") and rel:
            rel = {c for c in rel if corpus.get(c, {}).get("doc_id") == q["doc_id"]}
        is_metric = q.get("type", "metric") == "metric"
        # 非指标题：相关叶子的父块集合（章节覆盖验收口径，展开前保存）
        rel_parents = {corpus.get(c, {}).get("parent_id") for c in rel} - {None}
        # 指标题：不展开父块（对齐报告 0.83/0.93 口径——报告时相关块无 parent_id，
        # rel=叶子；重嵌入补 parent_id 后展开会稀释 NDCG）。非指标题保留父块展开（章节覆盖口径）。
        if not is_metric:
            rel = _expand_rel_with_parents(rel, corpus)
        if not rel:
            print(f"  [跳过] 相关集为空: {q['question'][:40]}")
            continue
        items.append({"q": q, "rel": rel, "is_metric": is_metric, "rel_parents": rel_parents})
    if args.only_non_metric:
        items = [it for it in items if not it["is_metric"]]
    if args.only_metric:
        items = [it for it in items if it["is_metric"]]
    print(f"有效条目: {len(items)}/{len(golden)}\n")

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()

    async def _one(item: dict) -> dict:
        async with sem:
            q = item["q"]
            raw_year = _extract_year(q["question"])
            hits = await _prod_real(
                q["question"], args.org, args.visibility, args.top_k, raw_year,
                full_router=args.full_router, rerank_candidates=args.rerank_candidates,
            )
            return {"q": q["question"], "ranked": [h["chunk_id"] for h in hits]}

    outs = await asyncio.gather(*[_one(it) for it in items])
    print(f"检索完成 {len(outs)} 题，耗时 {time.perf_counter() - t0:.1f}s\n")
    ranked_map = {o["q"]: o["ranked"] for o in outs}
    k_suffix = f"@{args.top_k}"

    # ---- 分组指标 ----
    def _group_metrics(sub: list[dict]) -> dict:
        ndcg = [_ndcg(ranked_map[it["q"]["question"]], it["rel"], args.top_k) for it in sub]
        mrr = [_mrr(ranked_map[it["q"]["question"]], it["rel"]) for it in sub]
        prec = [_prec(ranked_map[it["q"]["question"]], it["rel"], args.top_k) for it in sub]
        rec = [_recall(ranked_map[it["q"]["question"]], it["rel"], args.top_k) for it in sub]
        top1 = [int(bool(ranked_map[it["q"]["question"]]) and ranked_map[it["q"]["question"]][0] in it["rel"]) for it in sub]
        prel = [
            _recall(ranked_map[it["q"]["question"]], it["rel_parents"], args.top_k)
            for it in sub
        ]
        all_hit = [
            _all_hit(ranked_map[it["q"]["question"]], it["rel"], args.top_k)
            for it in sub
        ]
        feasible = [it for it in sub if len(it["rel"]) <= args.top_k]
        feasible_all_hit = [
            _all_hit(ranked_map[it["q"]["question"]], it["rel"], args.top_k)
            for it in feasible
        ]
        parent_sets = [it for it in sub if it["rel_parents"]]
        parent_all_hit = [
            _all_hit(ranked_map[it["q"]["question"]], it["rel_parents"], args.top_k)
            for it in parent_sets
        ]
        return {
            "n": len(sub),
            f"ndcg{k_suffix}": sum(ndcg) / len(ndcg),
            "mrr": sum(mrr) / len(mrr),
            f"prec{k_suffix}": sum(prec) / len(prec),
            f"rec{k_suffix}": sum(rec) / len(rec),
            "top1_hit": sum(top1) / len(top1),
            "ndcg@1": sum(top1) / len(top1),
            "prec@1": sum(top1) / len(top1),
            "rec@1": sum(
                _recall(ranked_map[it["q"]["question"]], it["rel"], 1) for it in sub
            ) / len(sub),
            f"parent_rec{k_suffix}": sum(prel) / len(prel),
            f"all_hit{k_suffix}": sum(all_hit) / len(all_hit),
            f"all_hit_feasible{k_suffix}": (
                sum(feasible_all_hit) / len(feasible_all_hit) if feasible_all_hit else 0.0
            ),
            "all_hit_feasible_n": len(feasible),
            f"parent_all_hit{k_suffix}": (
                sum(parent_all_hit) / len(parent_all_hit) if parent_all_hit else 0.0
            ),
        }

    metric_items = [it for it in items if it["is_metric"]]
    non_metric_items = [it for it in items if not it["is_metric"]]
    m = _group_metrics(metric_items) if metric_items else None
    nm = _group_metrics(non_metric_items) if non_metric_items else None

    # 指标题三个抽取准确率（本地确定性，无检索依赖）
    ext = [_metric_extraction(q["q"], args.org, args.visibility) for q in metric_items] if metric_items else []
    n_year = sum(1 for q in metric_items if q["q"].get("year") is not None) if metric_items else 0
    subject_acc = sum(1 for x in ext if x["subject_ok"]) / len(ext) if ext else 0.0
    year_acc = sum(1 for x in ext if x["year_ok"]) / n_year if n_year else 0.0
    value_acc = sum(1 for x in ext if x["value_ok"]) / len(ext) if ext else 0.0

    if m:
        print("=== 指标题（n=%d）— 验收：答案正确性（relay 保底，抓对即可） ===" % m["n"])
        print(
            f"  NDCG@1 {m['ndcg@1']:.4f} | Prec@1 {m['prec@1']:.4f} | "
            f"Rec@1 {m['rec@1']:.4f} | MRR {m['mrr']:.4f}"
        )
        print(f"  科目同义匹配准确率: {subject_acc:.4f} ({sum(1 for x in ext if x['subject_ok'])}/{len(ext)})")
        print(f"  时间周期准确率:     {year_acc:.4f} ({sum(1 for x in ext if x['year_ok'])}/{n_year})")
        print(f"  财务数值提取准确率: {value_acc:.4f} ({sum(1 for x in ext if x['value_ok'])}/{len(ext)})")
        print(
            f"  [参考] NDCG{k_suffix} {m[f'ndcg{k_suffix}']:.4f} | "
            f"Prec{k_suffix} {m[f'prec{k_suffix}']:.4f} | Rec{k_suffix} {m[f'rec{k_suffix}']:.4f}"
        )
        print()
    if nm:
        print("=== 非指标题（n=%d）— 验收：章节覆盖 + 召回排序 ===" % nm["n"])
        print(f"  父块Rec{k_suffix}（章节覆盖） {nm[f'parent_rec{k_suffix}']:.4f}")
        print(
            f"  NDCG{k_suffix} {nm[f'ndcg{k_suffix}']:.4f} | "
            f"Rec{k_suffix} {nm[f'rec{k_suffix}']:.4f} | MRR {nm['mrr']:.4f} | "
            f"Prec{k_suffix} {nm[f'prec{k_suffix}']:.4f}"
        )
        print(
            f"  All-Hit{k_suffix} {nm[f'all_hit{k_suffix}']:.4f} | "
            f"可达子集 All-Hit{k_suffix} {nm[f'all_hit_feasible{k_suffix}']:.4f} "
            f"({nm['all_hit_feasible_n']}/{nm['n']}) | "
            f"父章节 All-Hit{k_suffix} {nm[f'parent_all_hit{k_suffix}']:.4f}"
        )

    # 落库
    metrics = {
        "top_k": args.top_k, "org": args.org, "n_questions": len(items),
        "metric": {k: round(v, 4) for k, v in m.items()} if m else None,
        "non_metric": {k: round(v, 4) for k, v in nm.items()} if nm else None,
        "extraction": {
            "subject_acc": round(subject_acc, 4), "year_acc": round(year_acc, 4),
            "value_acc": round(value_acc, 4), "n_metric": len(ext), "n_year": n_year,
        },
        "summary": args.summary,
    }
    if args.dump:
        dump = {
            "metric": {it["q"]["question"]: _ndcg(ranked_map[it["q"]["question"]], it["rel"], args.top_k) for it in metric_items} if metric_items else {},
            "metric_mrr": {it["q"]["question"]: _mrr(ranked_map[it["q"]["question"]], it["rel"]) for it in metric_items} if metric_items else {},
            "non_metric": {it["q"]["question"]: _ndcg(ranked_map[it["q"]["question"]], it["rel"], args.top_k) for it in non_metric_items},
            "mrr_non_metric": {it["q"]["question"]: _mrr(ranked_map[it["q"]["question"]], it["rel"]) for it in non_metric_items},
        }
        Path(args.dump).write_text(json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n逐题 NDCG 已写入 {args.dump}")
    try:
        rec = save_eval_result("metrics", f"org={args.org};top_k={args.top_k};summary={args.summary}", metrics, {})
        print(f"\n已写入 eval_results: id={rec.id}")
    except Exception as e:  # noqa: BLE001
        print(f"写入 eval_results 失败（忽略）: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
