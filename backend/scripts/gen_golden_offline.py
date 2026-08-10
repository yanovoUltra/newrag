"""自动生成离线评测 Golden Dataset：从 financial_fields 索引生成问答对并标注相关 Chunk ID。

设计（离线指标定量评测需要"标注好的测试集"）：
- 语料：financial_fields 表（value/raw/unit/metric_label/year/company/doc_id/page）
- 每题：{question, snippet, unit, doc_id, metric, year, relevant_chunk_ids}
- 相关集标注：在 Qdrant chunks 中，定位"同一 doc + 归一化内容同时含 (指标标签, 数值)"的 chunk，
  视为该问题的正确答案所在 chunk；找不到则跳过该题（避免脏标注）。
- 输出：scripts/golden/offline_257.json（被 eval_ablation / eval_retrieval / eval_fields / eval_idf 等评测脚本共用）

用法（backend/ 下）：python scripts/gen_golden_offline.py [--max 200] [--org default]
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

from sqlalchemy import select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.models.entities import FinancialField  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402
from app.store.registry import get_session, init_db  # noqa: E402

GOLDEN_DIR = SCRIPT_DIR / "golden"
OUT_FILE = GOLDEN_DIR / "offline_257.json"

# 排除的测试/样例文档（非正式财报，避免污染标注集）
_EXCLUDE_DOC = {"_scan_test_茅台5页.pdf", "sample_report.pdf"}

# 中文问题模板：metric key -> 问句补全（"XX<year>年YY是多少？"）
_CN_Q = {
    "revenue": "{c}{y}年营业收入是多少？",
    "net_profit": "{c}{y}年归属于母公司股东的净利润是多少？",
    "eps": "{c}{y}年基本每股收益是多少？",
    "roe": "{c}{y}年加权平均净资产收益率是多少？",
    "roa": "{c}{y}年总资产收益率是多少？",
    "gross_margin": "{c}{y}年毛利率是多少？",
    "net_margin": "{c}{y}年净利率是多少？",
    "debt_ratio": "{c}{y}年资产负债率是多少？",
    "operating_cashflow": "{c}{y}年经营活动现金流量净额是多少？",
    "total_assets": "{c}{y}年总资产是多少？",
    "net_assets": "{c}{y}年净资产是多少？",
    "gross_profit": "{c}{y}年营业利润是多少？",
    "total_profit": "{c}{y}年利润总额是多少？",
}

# 英文问题模板（Apple/TSLA 等英文财报）
_EN_Q = {
    "revenue": "What was {c}'s total net sales in fiscal year {y}?",
    "net_profit": "What was {c}'s net income in fiscal year {y}?",
    "eps": "What was {c}'s basic earnings per share in fiscal year {y}?",
    "roe": "What was {c}'s return on equity in fiscal year {y}?",
    "roa": "What was {c}'s return on assets in fiscal year {y}?",
    "gross_margin": "What was {c}'s gross margin in fiscal year {y}?",
    "net_margin": "What was {c}'s net margin in fiscal year {y}?",
    "debt_ratio": "What was {c}'s debt-to-asset ratio in fiscal year {y}?",
    "operating_cashflow": "What was {c}'s net cash from operating activities in fiscal year {y}?",
    "total_assets": "What was {c}'s total assets in fiscal year {y}?",
    "net_assets": "What was {c}'s total shareholders' equity in fiscal year {y}?",
    "gross_profit": "What was {c}'s operating income in fiscal year {y}?",
    "total_profit": "What was {c}'s total profit in fiscal year {y}?",
}


def _norm(text) -> str:
    return (
        text.replace(",", "")
        .replace("，", "")
        .replace(" ", "")
        .replace("\u00a0", "")
        .replace("\n", "")
    )


def _load_corpus() -> dict[str, dict]:
    """滚动全量 payload：chunk_id -> {doc_id, norm, doc_name, chunk_type}。"""
    client = qdrant_store.get_client()
    settings = get_settings()
    corpus: dict[str, dict] = {}
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
            pay = p.payload or {}
            cid = pay.get("chunk_id")
            if not cid:
                continue
            corpus[cid] = {
                "doc_id": pay.get("doc_id") or "",
                "doc_name": pay.get("doc_name") or "",
                "norm": _norm(pay.get("content") or ""),
                "chunk_type": pay.get("chunk_type", "text"),
            }
        if res[1] is None:
            break
        offset = res[1]
    return corpus


def _load_fields(org_id: str) -> list[FinancialField]:
    with get_session() as s:
        return list(
            s.scalars(
                select(FinancialField)
                .where(FinancialField.org_id == org_id)
                .order_by(FinancialField.doc_id, FinancialField.year, FinancialField.metric)
            ).all()
        )


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _make_question(f: FinancialField) -> str | None:
    templates = _EN_Q if _is_ascii(f.company) else _CN_Q
    tpl = templates.get(f.metric)
    if not tpl:
        return None
    c = f.company.strip()
    return tpl.format(c=c, y=f.year)


def main() -> int:
    parser = argparse.ArgumentParser(description="自动生成离线评测标注集")
    parser.add_argument("--max", type=int, default=200, help="最多生成问答对数")
    parser.add_argument("--org", default="default")
    args = parser.parse_args()

    init_db()

    # 排除测试/样例文档的 doc_id
    settings = get_settings()
    from app.models.entities import Document

    with get_session() as s:
        docs = {
            d.id: d.filename
            for d in s.scalars(select(Document).where(Document.status == "indexed")).all()
        }
    excluded = {did for did, fn in docs.items() if fn in _EXCLUDE_DOC}
    print(f"排除测试/样例文档 {len(excluded)} 个；共 {len(docs)} 个 indexed 文档")

    print("加载语料（滚动全量 payload）...")
    corpus = _load_corpus()
    print(f"语料 chunk 数: {len(corpus)}")
    # doc_id 索引以加速定位：cid -> (norm, chunk_type)
    by_doc: dict[str, list[tuple[str, str, str]]] = {}
    for cid, info in corpus.items():
        by_doc.setdefault(info["doc_id"], []).append((cid, info["norm"], info["chunk_type"]))

    fields = _load_fields(args.org)
    print(f"字段总数: {len(fields)}")

    questions: list[dict] = []
    skipped = {"no_doc": 0, "no_tpl": 0, "no_rel": 0}
    seen = set()
    for f in fields:
        if len(questions) >= args.max:
            break
        if f.doc_id in excluded:
            continue
        q = _make_question(f)
        if not q:
            skipped["no_tpl"] += 1
            continue
        norm_raw = _norm(f.raw)
        norm_label = _norm(f.metric_label)
        candidates = by_doc.get(f.doc_id) or []
        if not candidates:
            skipped["no_doc"] += 1
            continue
        # 相关集：同 doc 且内容同时含 (指标标签, 数值) 的叶子块（排除 section 父块——检索端不召回）
        rel = [
            cid for cid, norm, ctype in candidates
            if ctype != "section"
            and norm_raw and norm_raw in norm and norm_label and norm_label in norm
        ]
        if not rel:
            # 兜底：仅含数值的叶子块（大表别列/文本块）
            rel = [cid for cid, norm, ctype in candidates
                   if ctype != "section" and norm_raw and norm_raw in norm]
        if not rel:
            skipped["no_rel"] += 1
            continue
        key = (f.doc_id, f.metric, f.year)
        if key in seen:
            continue
        seen.add(key)
        questions.append(
            {
                "question": q,
                "snippet": f.raw,
                "unit": f.unit or "",
                "doc_id": f.doc_id,
                "doc_name": docs.get(f.doc_id, ""),
                "metric": f.metric,
                "metric_label": f.metric_label,
                "year": f.year,
                "company": f.company,
                "relevant_chunk_ids": rel,
            }
        )

    print(f"生成问答对: {len(questions)}（跳过: {skipped}）")
    if len(questions) < 100:
        print("警告：生成数量不足 100，建议核查字段/语料覆盖。")

    data = {
        "_说明": "离线检索评测集（自动生成）。relevant_chunk_ids 为'正确答案所在 chunk'（同 doc 内容含指标标签+数值的块）。指标：Recall@K/Precision@K/NDCG@K/MRR。",
        "usage": "python scripts/eval_ablation.py --top-k 8  (或 eval_retrieval.py / eval_fields.py / eval_idf.py)",
        "n": len(questions),
        "questions": questions,
    }
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入: {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())