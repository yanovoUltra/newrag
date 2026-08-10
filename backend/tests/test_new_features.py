"""新功能测试：语义精切、接地校验、会话记忆。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.generation.grounding import extract_numbers, ground_answer
from app.splitter.chunker import Chunk
from app.splitter.semantic import _split_at_valleys, _split_sentences, refine_leaves_semantically
from app.core.config import get_settings


# ---- 接地校验 ----

def test_extract_numbers_dedup_and_percent():
    nums = extract_numbers("营收1,707.48亿元，同比增长2%，净利率14.3%。1,707.48亿元。")
    assert "1,707.48" in nums
    assert "2%" in nums
    assert "14.3%" in nums
    # 去重保序
    assert nums.count("1,707.48") == 1


def test_ground_answer_found_and_missing():
    blocks = [{"content": "营业收入1,707.48亿元，归母净利润452.57亿元。", "parent_content": ""}]
    g = ground_answer("营业收入1,707.48亿元，归母净利润450亿元。", blocks)
    assert g["checked"] >= 2
    assert "1,707.48" not in g["missing"]
    # 450 不在知识块中 → 列为 missing
    assert "450" in g["missing"]


def test_ground_answer_empty():
    g = ground_answer("", [{"content": "x"}])
    assert g == {"checked": 0, "missing": [], "citations": 0}


# ---- 语义精切 ----

def test_split_sentences():
    assert _split_sentences("第一句。第二句！第三句？") == ["第一句。", "第二句！", "第三句？"]


def test_split_at_valleys():
    sentences = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
    # sims[i] = sim(句 i, 句 i+1)，sims[2]=0.1 表示 s3|s4 之间为主题切换点
    sims = [0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9]
    # min_tokens 下限用极小值，确保断点生效（生产值为 128）
    settings = SimpleNamespace(chunk_min_tokens=2, semantic_split_sim_threshold=0.5)
    parts = _split_at_valleys(sentences, sims, settings)
    assert len(parts) == 2
    assert parts[0] == "s1s2s3"
    assert parts[1] == "s4s5s6s7s8"


def test_refine_skips_mock_embedder():
    """mock 嵌入（测试环境默认）无真实语义，精切应原样返回。"""
    from app.embed.embedder import get_embedder

    assert get_embedder().name == "mock"
    settings = get_settings()
    leaf = Chunk(
        id="x1", doc_id="d1", page=1, section_path="一、概况",
        chunk_type="text", content="，" .join(["这是一段内容。" for _ in range(10)]),
        token_count=200, seq=1,
    )
    out = refine_leaves_semantically([leaf], settings)
    assert out == [leaf]


# ---- 会话记忆 ----

def test_session_roundtrip():
    import uuid

    from app.store.session import _redis, append_turn, get_history

    if _redis() is None:
        pytest.skip("Redis 不可用")
    sid = f"t_{uuid.uuid4().hex}"  # 每次运行唯一 key，避免历史残留
    append_turn(sid, "q1", "a1")
    append_turn(sid, "q2", "a2")
    history = get_history(sid)
    assert [h["role"] for h in history] == ["user", "assistant", "user", "assistant"]
    assert history[-2]["content"] == "q2"
    assert history[-1]["content"] == "a2"
    # 空 session_id 无副作用
    assert get_history("") == []


# ---- 数据清洗：NFKC 归一化 + 页眉页脚去重 ----

def test_clean_nfkc_normalizes_numbers():
    from app.parsers.clean import clean_layout, normalize_text
    from app.parsers.base import LayoutResult, ParsedPage

    assert normalize_text("营收１，７０７．４８亿元，增长１８．２％") == "营收1,707.48亿元,增长18.2%"
    layout = LayoutResult(
        pages=[ParsedPage(page_no=1, text="１００％全角"), ParsedPage(page_no=2, text="normal")],
        text_extraction_rate=1.0,
    )
    clean_layout(layout, SimpleNamespace(clean_enable=True, clean_nfkc=True, clean_header_footer=False))
    assert layout.pages[0].text == "100%全角"


def test_clean_strips_repeated_headers_only_at_edges():
    from app.parsers.clean import clean_layout
    from app.parsers.base import LayoutResult, ParsedPage

    header = "上海浦东发展银行股份有限公司 2024年年度报告"
    footer = "第 1 页 共 50 页"
    settings = SimpleNamespace(
        clean_enable=True, clean_nfkc=True, clean_header_footer=True,
        clean_header_footer_min_ratio=0.5, clean_header_footer_max_tokens=24,
        clean_header_footer_margin=0.15,
    )
    pages = []
    for i in range(1, 7):
        # 页眉在首行、页脚在末行、正文在中部；页脚行号含变化不重复
        body = f"第{i}节 正文内容这里是唯一的一行正文。"
        pages.append(ParsedPage(page_no=i, text=f"{header}\n{body}\n{footer.replace('1', str(i))}"))
    layout = LayoutResult(pages=pages, text_extraction_rate=1.0)
    clean_layout(layout, settings)
    for p in layout.pages:
        assert header not in p.text
        assert "正文内容" in p.text


# ---- 问题1：陈旧文档防护 ----

def test_extract_year():
    from app.pipelines.answer import _extract_year

    assert _extract_year("浦发银行2024年营业收入是多少？") == 2024
    assert _extract_year("近三年营收趋势如何？") is None
    assert _extract_year("What was Apple's net sales in fiscal year 2024?") == 2024


def test_same_file_key():
    from app.api.v1.documents import _same_file_key

    assert _same_file_key("600000_2024年报.PDF", "600000_2024年报.pdf")
    assert _same_file_key("aa.docx", "bb.docx") is False


# ---- 数据清洗：页码剔除 + 接地校验单位归一化 ----

def test_clean_strips_page_numbers_only_at_edges():
    from app.parsers.clean import clean_layout
    from app.parsers.base import LayoutResult, ParsedPage

    settings = SimpleNamespace(
        clean_enable=True, clean_nfkc=True, clean_header_footer=False,
        clean_header_footer_min_ratio=0.5, clean_header_footer_max_tokens=24,
        clean_header_footer_margin=0.15, clean_page_numbers=True,
    )
    # 页尾纯数字（页码）与页首 "第 2 页" 应剔除；页中部数字行保留
    pages = [
        ParsedPage(page_no=1, text="第 1 页\n正文第一行\n正文第二行\n正文第三行\n正文第四行\n正文第五行\n25"),
        ParsedPage(page_no=2, text="第 2 页\n正文内容包含数字 2024 是正常的\n正文第三行\n26"),
    ]
    layout = LayoutResult(pages=pages, text_extraction_rate=1.0)
    clean_layout(layout, settings)
    assert "第 1 页" not in layout.pages[0].text
    assert "\n25" not in layout.pages[0].text
    assert "第 2 页" not in layout.pages[1].text
    # 页中部数字行（如正文里的年份）不受影响
    assert "2024" in layout.pages[1].text


def test_clean_keeps_page_numbers_in_middle():
    from app.parsers.clean import _PAGE_NO_RE

    assert _PAGE_NO_RE.match("25")
    assert _PAGE_NO_RE.match("第 3 页")
    assert _PAGE_NO_RE.match("- 12 -")
    assert _PAGE_NO_RE.match("Page 7")
    assert not _PAGE_NO_RE.match("营业收入 2024 亿元")
    assert not _PAGE_NO_RE.match("25 0000")


def test_normalize_number_expr_units():
    from app.generation.grounding import normalize_number_expr

    exprs = normalize_number_expr("营业收入1,707,480万元，同比归母净利润1707.48亿元，每股收益1.28元。")
    vals = {round(v, 4) for v, _ in exprs}
    # 1,707,480 万元 = 1.70748e10 元；1707.48 亿元 = 1.70748e11 元（不同量级，不误并）
    assert 17074800000.0 in vals
    assert 170748000000.0 in vals
    assert 1.28 in vals
    # 无单位的普通数字不纳入（年份等）
    assert normalize_number_expr("2024年营收增长18.2%") == []


def test_ground_answer_cross_unit_match():
    """跨单位等价：回答"17.07亿元" 与 知识块"170,748万元" 应数值匹配（非 missing）。"""
    from app.generation.grounding import ground_answer

    blocks = [{"content": "营业收入170,748万元，同比增长3.4%。", "parent_content": ""}]
    g = ground_answer("营业收入17.07亿元，同比增长3.4%。", blocks)
    assert g["checked"] >= 2
    assert "17.07" not in g["missing"]
    assert "3.4" not in g["missing"]
    # 单位不同但数值不等 → 仍应列为 missing
    g2 = ground_answer("营业收入18.5亿元。", blocks)
    assert "18.5" in g2["missing"]
