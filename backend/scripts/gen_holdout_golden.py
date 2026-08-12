"""holdout 指标题 golden 生成器：从 financial_fields 未覆盖组合采样，避免与 eval_set 重复。

- 数据源：financial_fields 表（(公司, 指标, 年份) 组合）+ Qdrant source_chunk_id 内容指纹
- 排除：eval_set.json 已用组合 + source_chunk_id 为空的行
- 题目模板：中文 "{公司}{年}年{指标标签}是多少？"；英文
  "What was {公司}'s {英文指标} in [fiscal year ]{年}?"（GE 系日历年用 in 1998，其余 fiscal year）
- snippet 来自字段 value（千分位格式化，与 eval_set 一致）
- 用法（backend/ 下）：.venv\\Scripts\\python.exe scripts\\gen_holdout_golden.py [--n 40] [--seed 7] [--out scripts/golden/holdout_40.json]
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402

import golden_utils  # noqa: E402

# 英文指标表述（与 eval_set 现有英文题风格一致，extract 可匹配）
_EN_METRIC = {
    "revenue": "total net sales",
    "net_profit": "net income",
    "eps": "basic earnings per share",
    "roa": "return on assets",
    "total_assets": "total assets",
    "net_assets": "total shareholders' equity",
    "gross_margin": "gross margin",
    "gross_profit": "operating income",
    "total_profit": "total profit",
    "operating_cashflow": "net cash provided by operating activities",
    "segment_revenues": "segment revenues",
}
# GE 系日历年文档（问题用 "in 1998"，其余英文用 "in fiscal year YYYY"）
_CALENDAR_YEAR_EN = {"General Electric", "GECS", "Power Systems"}


def _fmt_value(value) -> str:
    """数字 → 千分位字符串（与 eval_set snippet 格式一致）。"""
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f == int(f):
        return f"{int(f):,}"
    return f"{f:,}"


def _question(company: str, metric_key: str, metric_label: str, year: int) -> str:
    if company and company[0].isascii():
        en = _EN_METRIC.get(metric_key, metric_label)
        if company in _CALENDAR_YEAR_EN:
            return f"What was {company}'s {en} in {year}?"
        return f"What was {company}'s {en} in fiscal year {year}?"
    return f"{company}{year}年{metric_label}是多少？"


def main() -> int:
    parser = argparse.ArgumentParser(description="holdout 指标题 golden 生成器")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    parser.add_argument("--eval-set", default=None, help="排除已用组合的基准集（默认 eval_set.json）")
    args = parser.parse_args()

    settings = get_settings()
    db_path = settings.resolved_registry_db
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "select company, metric, metric_label, year, value, unit, doc_id, source_chunk_id "
        "from financial_fields where source_chunk_id is not null and source_chunk_id != ''"
    ).fetchall()
    con.close()
    print(f"字段表有效行（有 src）: {len(rows)}")

    eval_set = json.loads(
        (Path(args.eval_set) if args.eval_set else SCRIPT_DIR / "golden" / "eval_set.json")
        .read_text(encoding="utf-8")
    )
    used = {
        (q.get("company"), q.get("metric"), q.get("year"))
        for q in eval_set["questions"]
        if q.get("type", "metric") == "metric"
    }
    cand = [r for r in rows if (r[0], r[1], r[3]) not in used]
    print(f"未覆盖组合: {len(cand)}（排除 eval_set 已用 {len(used)}）")

    rng = random.Random(args.seed)
    if len(cand) < args.n:
        print(f"[警告] 候选不足 {args.n}，实际生成 {len(cand)}")
    chosen = rng.sample(cand, min(args.n, len(cand)))

    # 按 doc 分组批量拉 content（计算指纹）
    by_doc: dict[str, list[str]] = {}
    for r in chosen:
        by_doc.setdefault(r[6], []).append(r[7])
    payloads: dict[str, dict] = {}
    for doc_id, cids in by_doc.items():
        payloads.update(qdrant_store.fetch_payloads(doc_id, list(dict.fromkeys(cids))))

    questions: list[dict] = []
    for company, metric, label, year, value, unit, doc_id, src_id in chosen:
        content = payloads.get(src_id, {}).get("content")
        if not content:
            continue
        questions.append({
            "question": _question(company, metric, label, year),
            "snippet": _fmt_value(value),
            "unit": unit or "",
            "doc_id": doc_id,
            "doc_name": payloads.get(src_id, {}).get("doc_name", ""),
            "metric": metric,
            "metric_label": label,
            "year": year,
            "company": company,
            "relevant_chunk_ids": [src_id],
            "relevant_fingerprints": [golden_utils.content_fingerprint(content)],
        })
    print(f"生成有效题: {len(questions)}/{len(chosen)}")
    for q in questions:
        print(f"  {q['question']}  | {q['company']} {q['metric']} {q['year']} = {q['snippet']}")

    out_path = Path(args.out) if args.out else SCRIPT_DIR / "golden" / f"holdout_{args.n}.json"
    doc = {
        "_说明": f"holdout 指标题集（{args.n} 道，seed={args.seed}）：从 financial_fields 未覆盖"
                f"（公司×指标×年份 不在 eval_set 已用 168 组合内）组合采样；snippet 来自字段值；"
                f"相关块=source_chunk_id 内容指纹（语料重建免疫）。用于验证指标题检索/relay 泛化。",
        "usage": f"python scripts/eval_metrics.py --golden scripts/golden/{out_path.name}",
        "n": len(questions),
        "questions": questions,
    }
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
