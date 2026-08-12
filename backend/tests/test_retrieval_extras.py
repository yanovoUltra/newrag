"""六/七/八 模块单测：数值范围约束、章节类型权重、对比题实体均衡。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.fields.numeric import numeric_match_score, parse_numeric_constraints  # noqa: E402
from app.pipelines.answer import (  # noqa: E402
    _company_core_keys,
    _detect_comparison_entities,
    _fallback_levels,
    _needs_fallback,
    _real_hits,
    _rebalance_comparison,
)
from app.retrieval.search import _section_weight  # noqa: E402

COMPANIES = ["中国工商银行", "招商銀行", "贵州茅台", "宜宾五粮液", "比亚迪", "宁德时代新能源"]


# ---------- 六：数值范围约束 ----------

def test_constraint_npl_below():
    cons = parse_numeric_constraints("不良贷款率低于1%吗？")
    assert [(c["op"], c["value"], c["family"]) for c in cons] == [("<", 1.0, "percent")]


def test_constraint_revenue_over_yi():
    cons = parse_numeric_constraints("营业收入超千亿？")
    assert [(c["op"], c["value"], c["family"]) for c in cons] == [(">", 1e11, "money")]


def test_constraint_not_below():
    cons = parse_numeric_constraints("不良贷款率不低于1.5%")
    assert [(c["op"], c["value"]) for c in cons] == [(">=", 1.5)]


def test_constraint_none():
    assert parse_numeric_constraints("为什么营收增长？") == []


def test_match_score_percent():
    c = [{"metric": "npl_ratio", "op": "<", "value": 1.0, "family": "percent"}]
    assert numeric_match_score("不良贷款率0.85%，较上年下降", c) == 1.0
    assert numeric_match_score("不良贷款率1.36%", c) == 0.0


def test_match_score_money():
    c = [{"metric": "revenue", "op": ">", "value": 1e11, "family": "money"}]
    assert numeric_match_score("全年营业收入1.2万亿元", c) == 1.0
    assert numeric_match_score("营业收入673.2亿元", c) == 0.0


# ---------- 七：章节类型权重 ----------

def test_section_weight():
    assert _section_weight({"section_path": "三、财务报表", "doc_name": "x.pdf"}) == 1.2
    assert _section_weight({"section_path": "管理层讨论与分析", "doc_name": "x.pdf"}) == 1.1
    assert _section_weight({"section_path": "目录", "doc_name": "x.pdf"}) == 0.7
    assert _section_weight({"section_path": "某普通章节", "doc_name": "x.pdf"}) == 1.0


# ---------- 八：对比题实体均衡保活 ----------

def test_company_core_keys():
    assert _company_core_keys("中国工商银行") == {"中国工商银行", "工商银行"}
    assert _company_core_keys("招商銀行") == {"招商银行"}  # 繁体归一


def test_detect_comparison_icbc_cmb():
    ents = _detect_comparison_entities("工商银行和招商银行的净利润对比？", COMPANIES)
    assert sorted(ents) == ["中国工商银行", "招商銀行"]


def test_detect_comparison_moutai_wuliangye():
    ents = _detect_comparison_entities("贵州茅台和五粮液哪个增速更高？", COMPANIES)
    assert sorted(ents) == ["宜宾五粮液", "贵州茅台"]


def test_detect_non_comparison():
    assert _detect_comparison_entities("2025年工商银行净利润是多少？", COMPANIES) == []


def test_rebalance_keeps_both_entities():
    hits = [{"chunk_id": f"icbc{i}", "content": "中国工商银行 净利润", "fused_score": 9 - i}
            for i in range(8)]
    hits += [{"chunk_id": f"cmb{i}", "content": "招商银行 净利润", "fused_score": 5 - i}
             for i in range(4)]
    rb = _rebalance_comparison(hits, ["中国工商银行", "招商銀行"], 8)
    assert len(rb) == 8
    assert len({h["chunk_id"] for h in rb}) == 8  # 无重复


# ---------- 十一：分级降级兜底 ----------

def test_real_hits_excludes_injected():
    hits = [
        {"chunk_id": "a", "is_relay": True},
        {"chunk_id": "b", "is_exact_phrase": True},
        {"chunk_id": "c", "chunk_type": "section"},
        {"chunk_id": "d", "chunk_type": "text"},
    ]
    assert _real_hits(hits) == 1


def test_needs_fallback_triggers():
    low = [{"chunk_id": "a", "chunk_type": "text"}, {"chunk_id": "b", "chunk_type": "text"}]
    assert _needs_fallback(low, threshold=2, min_candidates=6)  # real_hits 0 < 2
    assert _needs_fallback([], threshold=2, min_candidates=6)    # 空
    short = [{"chunk_id": f"x{i}", "chunk_type": "text"} for i in range(2)]
    assert _needs_fallback(short, threshold=1, min_candidates=6)  # 候选不足


def test_needs_fallback_exemptions():
    relay = [{"chunk_id": "a", "is_relay": True}, {"chunk_id": "b", "chunk_type": "text"}]
    assert not _needs_fallback(relay, threshold=2, min_candidates=6)  # relay 保底豁免
    parent = [{"chunk_id": "a", "chunk_type": "section"},
              {"chunk_id": "b", "chunk_type": "section"},
              {"chunk_id": "c", "chunk_type": "text"}]
    assert not _needs_fallback(parent, threshold=2, min_candidates=6)  # 综述父块豁免


def test_fallback_levels_years():
    levels = _fallback_levels(fy=2025, window=1)
    assert [lv for lv, _ in levels] == [1, 2, 2, 3, 4]  # L2 两个子步
    assert levels[1][1]["fiscal_years"] == [2024, 2025, 2026]  # L2 前后年窗口
    assert levels[2][1]["fiscal_years"] is None                 # L2 无年份
    assert levels[3][1]["use_subject_filter"] is False          # L3 关实体
    assert levels[4][1]["dense_only"] is True                   # L4 纯 dense
    # 逐级继承：后级包含前级放宽
    assert levels[4][1]["use_phrase_hard_insert"] is False


def test_fallback_levels_no_year():
    levels = _fallback_levels(fy=None, window=1)
    assert [lv for lv, _ in levels] == [1, 2, 3, 4]  # 趋势已放宽，L2 只留无年份子步
