"""字段抽取检索消融：改动前（仅混合检索）vs 改动后（混合检索 + 字段抽取）。

度量：答案取值命中率——golden 中正确数字片段（归一化）是否出现在 surface 证据中。
- baseline（改动前）：hybrid_search top-k 的块内容（content + 父块 content）。
- +fields（改动后）：metric 类问题追加字段抽取证据（raw 值），再叠加 top-k 块内容。

用途：判断字段抽取相对改动前是正收益还是负收益。
- 正收益：字段命中让"改动后"多命中 → 见 per-question 中 baseline_hit=0 且 fields_hit=1 的行。
- 噪声：fields 返回了值但未命中（fields_found=1 且 fields_hit=0）→ 可能误导，需人工核验。

结果写入 eval_results 表（eval_name=ablation_fields）。
用法（backend/ 下）：python scripts/eval_fields.py [--top-k 8] [--org default] [--visibility public]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.embed.embedder import get_embedder  # noqa: E402
from app.fields.metrics import extract_metric_from_question, parse_number, unit_to_yuan  # noqa: E402
from app.fields.subject import find_subject_company  # noqa: E402
from app.pipelines.answer import _lookup_field_evidence  # noqa: E402
from app.retrieval.router import classify_query_type_keyword, route_query  # noqa: E402
from app.retrieval.search import hybrid_search  # noqa: E402
from app.store.registry import get_document, init_db, list_field_companies, query_fields, save_eval_result  # noqa: E402
from eval_ablation import _norm  # noqa: E402

GOLDEN_FILE = SCRIPT_DIR / "golden" / "offline_257.json"


def _block_texts(hits: list[dict]) -> list[str]:
    texts = []
    for h in hits:
        c = h.get("content") or ""
        if c:
            texts.append(c)
        p = h.get("parent_content") or ""
        if p:
            texts.append(p)
    return texts


def _hit(texts: list[str], snippet_n: str) -> bool:
    return any(snippet_n in _norm(t) for t in texts)


def _snippet_target(snippet: str) -> tuple[str, float | None]:
    """从 snippet 提取：基线块匹配用完整归一化串 + 字段匹配用数值目标。

    EPS/ROE 等复合 snippet 形如"指标标签|数值"，取最后一段 | 后的数字作为字段数值目标；
    其它纯数字 snippet 直接取数值。
    """
    sn = _norm(snippet)
    tail = snippet.rsplit("|", 1)[-1] if "|" in snippet else snippet
    v, _ = parse_number(tail)
    return sn, v


def _field_dict(f) -> dict:
    """把 FinancialField ORM 行转成统一 dict（与 _lookup_field_evidence 输出对齐）。"""
    doc = get_document(f.doc_id)
    return {
        "metric_label": f.metric_label,
        "year": f.year,
        "value": f.value,
        "raw": f.raw,
        "doc_name": doc.filename if doc else f.doc_id,
        "company": f.company,
    }


def _fields_hit(fields: list[dict], sn: str, target: float | None, unit: str = "") -> bool:
    """字段命中：单位感知比较——字段值与 golden 目标各自折算为"元"后精确相等，
    或 raw 含数值目标；无数值目标时用归一化串包含。比率/EPS 类单位未知视为 ×1。
    """
    if target is not None:
        target_yuan = unit_to_yuan(target, unit)
        return any(
            f.get("value") is not None
            and (abs(unit_to_yuan(f["value"], f.get("unit")) - target_yuan) < 1e-6
                 or _norm(str(f["raw"])) == _norm(str(target)))
            for f in fields
        )
    return any(sn in _norm(f["raw"]) for f in fields)


def main() -> int:
    parser = argparse.ArgumentParser(description="字段抽取 vs 改动前混合检索（取值命中率）")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--org", default="default")
    parser.add_argument("--visibility", default="public")
    args = parser.parse_args()

    settings = get_settings()
    init_db()
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))["questions"]

    embedder = get_embedder()
    rows = []
    for q in golden:
        question, snippet = q["question"], q["snippet"]
        qunit = q.get("unit", "")
        sn, target = _snippet_target(snippet)
        # 与生产一致：路由给出 query_type（metric 才会走字段抽取）
        try:
            import asyncio

            plan = asyncio.run(route_query(question))
        except Exception:
            plan = None
        qtype = plan.query_type if plan else classify_query_type_keyword(question)

        dense, sparse = embedder.embed_texts_with_sparse([question], text_type="query")
        sparse_vec = sparse[0] if sparse else None
        if sparse_vec and qtype == "metric":
            from app.retrieval.idf import neutralize_idf

            sparse_vec = neutralize_idf(sparse_vec)
        hits = hybrid_search(
            query_vector=dense[0], org_id=args.org, user_visibility=args.visibility,
            query_text=question, top_k=args.top_k, query_sparse=sparse_vec,
        )
        base_texts = _block_texts(hits)
        baseline_hit = _hit(base_texts, sn)

        fields_before: list[dict] = []  # org-wide 无条件（改动前，跨公司混合）
        fields_after: list[dict] = []   # 主体公司过滤（改动后）
        fields_before_hit = False
        fields_after_hit = False
        subject: str | None = None
        if qtype == "metric":
            key, q_year = extract_metric_from_question(question)
            subject = find_subject_company(question, list_field_companies(args.org, args.visibility))
            if key is not None:
                fields_before = [_field_dict(r) for r in query_fields(
                    [key], args.org, args.visibility, year=q_year, limit=20)]
            fields_after = _lookup_field_evidence(question, args.org, args.visibility, None)
            fields_before_hit = _fields_hit(fields_before, sn, target, qunit)
            fields_after_hit = _fields_hit(fields_after, sn, target, qunit)
        changed_before = baseline_hit or fields_before_hit
        changed_after = baseline_hit or fields_after_hit

        rows.append(
            {
                "question": question,
                "snippet": snippet,
                "type": qtype,
                "subject": subject,
                "baseline_hit": baseline_hit,
                "fields_before_found": bool(fields_before),
                "fields_before_hit": fields_before_hit,
                "fields_after_found": bool(fields_after),
                "fields_after_hit": fields_after_hit,
                "changed_before": changed_before,
                "changed_after": changed_after,
                "fields_before": [
                    {"label": f["metric_label"], "year": f["year"], "raw": f["raw"], "company": f["company"], "doc": f["doc_name"]}
                    for f in fields_before
                ],
                "fields_after": [
                    {"label": f["metric_label"], "year": f["year"], "raw": f["raw"], "company": f["company"], "doc": f["doc_name"]}
                    for f in fields_after
                ],
            }
        )

    n = len(rows)
    base_hits = sum(r["baseline_hit"] for r in rows)          # 仅混合检索
    before_hits = sum(r["changed_before"] for r in rows)      # 混合检索 + org-wide 字段
    after_hits = sum(r["changed_after"] for r in rows)        # 混合检索 + 主体过滤字段
    before_field_hits = sum(r["fields_before_hit"] for r in rows)
    after_field_hits = sum(r["fields_after_hit"] for r in rows)
    noise_before = [r for r in rows if r["fields_before_found"] and not r["fields_before_hit"]]
    noise_after = [r for r in rows if r["fields_after_found"] and not r["fields_after_hit"]]
    added_after = [r for r in rows if not r["baseline_hit"] and r["changed_after"]]

    print(f"\n{'问题':<32}{'主体':<9}{'基态':<5}{'字段前':<6}{'字段后':<6}{'改前':<5}{'改后':<5}")
    print("-" * 80)
    for r in rows:
        print(
            f"{r['question'][:30]:<32}{str(r['subject'] or '-')[:8]:<9}{str(r['baseline_hit']):<5}"
            f"{str(r['fields_before_hit']):<6}{str(r['fields_after_hit']):<6}"
            f"{str(r['changed_before']):<5}{str(r['changed_after']):<5}"
        )

    print("\n=== 汇总 ===")
    print(f"条目数: {n}")
    print(f"改动前(仅混合检索)命中: {base_hits}/{n} = {base_hits/n:.1%}")
    print(f"改动后A(混合+org-wide字段)命中: {before_hits}/{n} = {before_hits/n:.1%}")
    print(f"改动后B(混合+主体过滤字段)命中: {after_hits}/{n} = {after_hits/n:.1%}")
    print(f"字段命中: 改动前A {before_field_hits}/{n} → 改动后B {after_field_hits}/{n}（净 {after_field_hits-before_field_hits:+d}）")
    print(f"噪声问题: 改动前A {len(noise_before)} → 改动后B {len(noise_after)}")
    if added_after:
        print("\n[字段正收益] 改动前未命中、靠字段抽取新增命中的问题（先于主体过滤，两种都命中）:")
        for r in added_after:
            print(f"  - {r['question']}  <- {r['snippet']}")
    if noise_before:
        print("\n[改动前A噪声] org-wide 字段返回了值但未命中:")
        for r in noise_before:
            print(f"  - {r['question']}  <- {r['snippet']}")
            for f in r["fields_before"]:
                print(f"      字段 {f['label']} {f['year']}年={f['raw']} ({f['company']}/{f['doc']})")
    if noise_after:
        print("\n[改动后B噪声] 主体过滤后仍返回了值但未命中:")
        for r in noise_after:
            print(f"  - {r['question']}  <- {r['snippet']}")
            for f in r["fields_after"]:
                print(f"      字段 {f['label']} {f['year']}年={f['raw']} ({f['company']}/{f['doc']})")

    metrics = {
        "top_k": args.top_k,
        "n": n,
        "hit_rate_baseline": round(base_hits / n, 4),
        "hit_rate_unscoped": round(before_hits / n, 4),
        "hit_rate_subject_scoped": round(after_hits / n, 4),
        "delta_subject_filter": after_hits - before_hits,
        "field_hits_unscoped": before_field_hits,
        "field_hits_subject_scoped": after_field_hits,
        "noise_unscoped": len(noise_before),
        "noise_subject_scoped": len(noise_after),
    }
    payload = {"conclusion": "主体过滤: hit_rate_unscoped vs subject_scoped; delta>0=正收益，noise 下降=更干净"}
    try:
        rec = save_eval_result(
            "ablation_fields", f"org={args.org};top_k={args.top_k}",
            metrics, {**payload, "per_question": [{k: v for k, v in r.items()} for r in rows]},
        )
        print(f"\n已写入 eval_results: id={rec.id}")
    except Exception as e:
        print(f"写入失败: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())