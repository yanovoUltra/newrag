"""生成对抗性/压力测试 golden：4 类"刁钻问题"边界用例，自动标注相关 Chunk ID。

目标：评测更贴近"通用检索"（自然语言问句），而非结构化指标问答。
类别（对应容易被击穿的漏洞）：
  A. 数值精度挑战：问题含精确数字，要求带精确数字的表格块排到 Top-3（不能只靠语义
     ——语义向量会把 123 亿和 123.5 亿混为一谈）；
  B. 否定词陷阱：'除了iPhone之外'，不能把 iPhone 独占块顶上来（应召回含整表的产品营收块）；
  C. 时间混淆：显式年份 → 必须通过 fiscal_year 过滤，不能混入其他年份文档块；
  D. 跨页长文本：MD&A 综述问题，要求分散在多个 chunk 的相关块全部召回，而非只抓第一段。

相关集定位：Qdrant payload 内容特征（同 doc + 关键词/章节前缀），自动滚动定位。

用法（backend/ 下）：python scripts/gen_adversarial_golden.py [--org default]
输出：scripts/golden/adversarial_v2.json（每类 5 题，共 20 题）
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.models.entities import Document  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402
from app.store.registry import get_session, init_db  # noqa: E402

GOLDEN_DIR = SCRIPT_DIR / "golden"
OUT_FILE = GOLDEN_DIR / "adversarial_v2.json"

# 用例定义：question + 相关集定位规则（相关集 = 同 doc 叶子块 content 满足条件）
# 定位规则键：
#   doc         文件名 glob（用于解析 doc_id）
#   must_all    内容须包含全部关键词（AND）
#   any         内容包含任一关键词（OR，与 must_all 组合：先 AND 后 OR）
#   section_prefix  section_path 前缀（如 '第三节 管理层讨论与分析'）
#   chunk_type  限定块类型（如 table）
#   leaf_only   排除 section 父块（检索端不召回，避免构造性虚低）
#   limit       相关集上限（相关块过多时截取，防评测偏向大相关集）
#   special     特殊检验标记（top3=数值精度须 Top-3 命中）
CASES = [
    # ---------- A. 数值精度挑战 ----------
    {
        "id": "A1_numeric_exact",
        "category": "numeric_precision",
        "question": "中芯国际2025年营业收入是67,323,192千元吗？",
        "doc": "688981_中芯国际*",
        "must_all": ["67,323,192"],
        "leaf_only": True,
        "special": "top3",
    },
    {
        "id": "A2_numeric_approx",
        "category": "numeric_precision",
        "question": "中芯国际2025年营业收入约为67,400,000千元吗？",
        "doc": "688981_中芯国际*",
        "must_all": ["67,323,192"],
        "leaf_only": True,
        "special": "top3",
    },
    {
        "id": "A3_numeric_npat",
        "category": "numeric_precision",
        "question": "浦发银行2024年归属于母公司股东的净利润是452.57亿元吗？",
        "doc": "600000_2024年报.pdf",
        "must_all": ["452.57"],
        "leaf_only": True,
        "special": "top3",
    },
    {
        "id": "A4_numeric_revenue",
        "category": "numeric_precision",
        "question": "浦发银行2024年营业收入是1,707.48亿元吗？",
        "doc": "600000_2024年报.pdf",
        "must_all": ["1,707.48"],
        "leaf_only": True,
        "special": "top3",
    },
    {
        "id": "A5_numeric_apple",
        "category": "numeric_precision",
        "question": "Apple在2024财年的净销售额是391,035百万美元吗？",
        "doc": "AAPL_10-K*",
        "must_all": ["391,035"],
        "leaf_only": True,
        "special": "top3",
    },
    # ---------- B. 否定词陷阱 ----------
    {
        "id": "B1_except_iphone",
        "category": "negation_trap",
        "question": "除了iPhone之外，Apple哪些产品的净销售额在增长？",
        "doc": "AAPL_10-K*",
        "must_all": ["iPhone", "Mac", "Net sales"],
        "chunk_type": "table",
        "special": "no_iphone_only",
    },
    {
        "id": "B2_mac_ipad",
        "category": "negation_trap",
        "question": "Apple的Mac和iPad净销售额在2024财年表现如何？",
        "doc": "AAPL_10-K*",
        "must_all": ["Mac", "Net sales"],
        "chunk_type": "table",
    },
    {
        "id": "B3_except_mac_decline",
        "category": "negation_trap",
        "question": "除了Mac之外，Apple哪些产品的净销售额在2024财年下降？",
        "doc": "AAPL_10-K*",
        "must_all": ["Mac", "Net sales"],
        "chunk_type": "table",
    },
    {
        "id": "B4_pufa_except_nii",
        "category": "negation_trap",
        "question": "浦发银行除了利息净收入之外，2024年还有哪些收入来源在增长？",
        "doc": "600000_2024年报.pdf",
        "must_all": ["利息净收入"],
        "leaf_only": True,
        "limit": 8,
    },
    {
        "id": "B5_byd_except_nev",
        "category": "negation_trap",
        "question": "除了新能源汽车之外，比亚迪还有哪些主营业务？",
        "doc": "002594_比亚迪*",
        "must_all": ["新能源汽车"],
        "leaf_only": True,
        "limit": 8,
    },
    # ---------- C. 时间混淆（年份过滤） ----------
    {
        "id": "C1_pufa_2024",
        "category": "year_confusion",
        "question": "浦发银行2024年营业收入是多少？",
        "doc": "600000_2024年报.pdf",
        "must_all": ["营业收入", "1,707.48"],
        "leaf_only": True,
        "special": "year_filter",
    },
    {
        "id": "C2_byd_2025",
        "category": "year_confusion",
        "question": "比亚迪2025年营业收入是多少？",
        "doc": "002594_比亚迪*",
        "must_all": ["营业收入"],
        "leaf_only": True,
        "limit": 10,
        "special": "year_filter",
    },
    {
        "id": "C3_smic_2025",
        "category": "year_confusion",
        "question": "中芯国际2025年营业收入是多少？",
        "doc": "688981_中芯国际*",
        "must_all": ["营业收入"],
        "leaf_only": True,
        "limit": 8,
        "special": "year_filter",
    },
    {
        "id": "C4_apple_2024",
        "category": "year_confusion",
        "question": "Apple在2024财年的净销售额是多少？",
        "doc": "AAPL_10-K*",
        "must_all": ["Net sales"],
        "leaf_only": True,
        "limit": 8,
        "special": "year_filter",
    },
    {
        "id": "C5_pufa_npat_2024",
        "category": "year_confusion",
        "question": "浦发银行2024年归属于母公司股东的净利润是多少？",
        "doc": "600000_2024年报.pdf",
        "must_all": ["归属于母公司股东的净利润"],
        "leaf_only": True,
        "limit": 8,
        "special": "year_filter",
    },
    # ---------- D. 跨页长文本（MD&A 多块召回） ----------
    {
        "id": "D1_mda_risk",
        "category": "long_text_crosspage",
        "question": "请总结浦发银行管理层讨论与分析（MD&A）中关于风险的观点",
        "doc": "600000_2024年报.pdf",
        "section_prefix": "第三节 管理层讨论与分析",
        "any": ["风险", "不良"],
        "leaf_only": True,
    },
    {
        "id": "D2_asset_quality",
        "category": "long_text_crosspage",
        "question": "浦发银行2024年对资产质量与不良贷款的管理情况如何？",
        "doc": "600000_2024年报.pdf",
        "section_prefix": "第三节 管理层讨论与分析",
        "must_all": ["不良"],
        "leaf_only": True,
    },
    {
        "id": "D3_main_business",
        "category": "long_text_crosspage",
        "question": "请总结浦发银行2024年报中主营业务发展情况的分析",
        "doc": "600000_2024年报.pdf",
        "must_all": ["主营业务"],
        "leaf_only": True,
        "limit": 6,
    },
    {
        "id": "D4_smic_rd",
        "category": "long_text_crosspage",
        "question": "请总结中芯国际2025年报中关于研发投入的情况",
        "doc": "688981_中芯国际*",
        "must_all": ["研发"],
        "leaf_only": True,
        "limit": 8,
    },
    {
        "id": "D5_byd_business",
        "category": "long_text_crosspage",
        "question": "请总结比亚迪2025年报中主营业务经营情况的分析",
        "doc": "002594_比亚迪*",
        "must_all": ["主营业务"],
        "leaf_only": True,
        "limit": 8,
    },
]


def _norm(text) -> str:
    return (
        str(text)
        .replace(",", "")
        .replace("，", "")
        .replace(" ", "")
        .replace("\u00a0", "")
        .replace("\n", "")
        .lower()
    )


def _resolve_doc_id(doc_glob: str, docs: dict[str, str]) -> str | None:
    for did, fn in docs.items():
        if fnmatch.fnmatch(fn, doc_glob):
            return did
    return None


def _locate_relevant(doc_id: str, rule: dict, corpus_by_doc: dict) -> list[str]:
    """按定位规则在 doc 内定位相关叶子块 chunk_id 列表。"""
    candidates = corpus_by_doc.get(doc_id) or []
    must_all = [_norm(k) for k in rule.get("must_all", [])]
    any_kw = [_norm(k) for k in rule.get("any", [])]
    section_prefix = rule.get("section_prefix")
    chunk_type = rule.get("chunk_type")
    leaf_only = rule.get("leaf_only", False)
    rel: list[str] = []
    for cid, norm, ctype, sp in candidates:
        if leaf_only and ctype == "section":
            continue
        if chunk_type and ctype != chunk_type:
            continue
        if section_prefix and not sp.startswith(section_prefix):
            continue
        if must_all and not all(k in norm for k in must_all):
            continue
        if any_kw and not any(k in norm for k in any_kw):
            continue
        rel.append(cid)
    limit = rule.get("limit")
    return rel[:limit] if limit else rel


def main() -> int:
    parser = argparse.ArgumentParser(description="生成对抗性评测标注集")
    parser.add_argument("--org", default="default")
    args = parser.parse_args()

    init_db()
    settings = get_settings()

    with get_session() as s:
        docs = {
            d.id: d.filename
            for d in s.scalars(select(Document).where(Document.status == "indexed")).all()
        }

    # 滚动全量 payload → 按 doc 分组 (chunk_id, 归一化内容, chunk_type, section_path)
    client = qdrant_store.get_client()
    by_doc: dict[str, list[tuple[str, str, str, str]]] = {}
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in pts:
            pay = p.payload or {}
            cid = pay.get("chunk_id")
            if not cid:
                continue
            by_doc.setdefault(pay.get("doc_id") or "", []).append(
                (cid, _norm(pay.get("content") or ""), pay.get("chunk_type", "text"),
                 str(pay.get("section_path") or ""))
            )
        if offset is None or not pts:
            break
    print(f"语料加载完成，{sum(len(v) for v in by_doc.values())} chunks / {len(by_doc)} docs")

    questions: list[dict] = []
    skipped: list[str] = []
    for rule in CASES:
        doc_id = _resolve_doc_id(rule["doc"], docs)
        if not doc_id:
            skipped.append(f"{rule['id']}: 文档未找到 {rule['doc']}")
            continue
        rel = _locate_relevant(doc_id, rule, by_doc)
        if not rel:
            skipped.append(f"{rule['id']}: 相关集为空（定位规则过严）")
            continue
        questions.append(
            {
                "id": rule["id"],
                "category": rule["category"],
                "question": rule["question"],
                "doc_id": doc_id,
                "doc_name": docs[doc_id],
                "special": rule.get("special", ""),
                "relevant_chunk_ids": rel,
            }
        )

    print(f"生成对抗性用例: {len(questions)}")
    if skipped:
        print("跳过:")
        for s_ in skipped:
            print("  -", s_)

    data = {
        "_说明": "对抗性/压力测试评测集 v2（通用检索场景，目录化 golden/）。4 类×5 题=20 题：数值精度(numeric_precision)/否定词(negation_trap)/时间混淆(year_confusion)/跨页长文本(long_text_crosspage)。",
        "usage": "python scripts/eval_adversarial.py --top-k 10 --k 5,8,10",
        "n": len(questions),
        "questions": questions,
    }
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入: {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
