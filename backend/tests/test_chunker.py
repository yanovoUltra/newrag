"""切分器单元测试。"""

from __future__ import annotations

from app.parsers.base import LayoutResult, ParsedPage, ParsedTable
from app.parsers.structure import Section, build_section_tree, section_path_for_page
from app.splitter.chunker import build_chunks
from app.splitter.tokens import count_tokens


def test_build_section_tree_and_path():
    layout = LayoutResult(
        pages=[
            ParsedPage(page_no=1, text="一、营业收入\n2023年营收86.5亿元。"),
            ParsedPage(page_no=2, text="二、净利润\n净利润7.9亿元。"),
        ]
    )
    sections = build_section_tree(layout)
    assert sections[0].title == "一、营业收入"
    assert sections[0].level == 1
    assert section_path_for_page(sections, 1) == "一、营业收入"
    assert section_path_for_page(sections, 2) == "二、净利润"


def test_chunks_text_and_table_types():
    layout = LayoutResult(
        pages=[
            ParsedPage(
                page_no=1,
                text="一、营业收入\n营收86.5亿元，同比增长18.2%。",
                tables=[
                    ParsedTable(page=1, headers=["季度", "营收"], rows=[["Q1", "18.9"], ["Q2", "21.3"]])
                ],
            )
        ]
    )
    chunks = build_chunks("doc1", layout, build_section_tree(layout), doc_title="t.pdf")
    types = [c.chunk_type for c in chunks]
    assert "text" in types and "table" in types
    table_chunk = next(c for c in chunks if c.chunk_type == "table")
    assert "Q1" in table_chunk.content and "|" in table_chunk.content  # 表格不可分割
    text_chunk = next(c for c in chunks if c.chunk_type == "text")
    assert text_chunk.section_path == "一、营业收入"
    assert text_chunk.doc_id == "doc1"


def test_chunk_token_bounds():
    long_para = "公司2023年实现营业收入86.5亿元，同比增长18.2%，毛利率29.6%。" * 40
    layout = LayoutResult(pages=[ParsedPage(page_no=1, text=f"三、概况\n{long_para}")])
    chunks = build_chunks("doc1", layout, build_section_tree(layout))
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    # 长段落被切分为多块（排除标题块）
    long_chunks = [c for c in text_chunks if c.token_count > 100]
    assert len(long_chunks) >= 2
    for c in text_chunks:
        assert c.token_count <= 256 + 2  # 叶子块不超过 max（含重叠允许小幅浮动）
    # 相邻块存在 10% 重叠字符
    assert long_chunks[0].content[-10:] in long_chunks[1].content


def test_count_tokens_smoke():
    assert count_tokens("中文文本测试") >= 1
    assert count_tokens("hello world") >= 1
