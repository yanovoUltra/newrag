"""切分器单元测试。"""

from __future__ import annotations

from app.parsers.base import LayoutResult, ParsedPage, ParsedTable
from app.parsers.structure import Section, build_section_tree, section_path_for_page
from app.splitter.chunker import Chunk, build_chunks
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


def test_text_lines_aggregated_and_noise_filtered():
    lines = [
        "报告期内公司实现营业收入，同比增长，营收结构持续优化。",
        "公司主营业务保持稳健，毛利率同比小幅提升。",
        "管理层对下一年度经营前景持审慎乐观态度。",
    ]
    text = "\n".join(["目录", "1", "第 3 页", "----------------", *lines])
    layout = LayoutResult(pages=[ParsedPage(page_no=1, text=text)])
    chunks = build_chunks("doc1", layout, build_section_tree(layout))
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    # 噪声行被过滤，正文行聚合进同一块（总 token 数未超 target）
    assert len(text_chunks) == 1
    content = text_chunks[0].content
    assert "目录" not in content
    assert "第 3 页" not in content
    assert "营收结构持续优化" in content
    assert "审慎乐观态度" in content


def test_parent_id_grouped_per_page():
    para = "公司主营业务保持稳健发展，收入结构与去年基本一致。"
    text = "\n".join(["一、概览"] + [para] * 16)  # 16 行 ≈ 400 token → 多个叶子块
    layout = LayoutResult(pages=[ParsedPage(page_no=1, text=text)])
    chunks = build_chunks("doc1", layout, build_section_tree(layout))
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    assert len(text_chunks) >= 2
    parents = {c.parent_id for c in text_chunks}
    assert len(parents) == 1  # 同页叶子共享同一父块
    assert None not in parents


def test_count_tokens_smoke():
    assert count_tokens("中文文本测试") >= 1
    assert count_tokens("hello world") >= 1


def test_split_text_by_tokens_single_line_overflow():
    from app.splitter.tokens import split_text_by_tokens

    # 无换行巨段：按字符硬切，任一块不超 max_tokens
    giant = "无标点长文本内容" * 500
    parts = split_text_by_tokens(giant, max_tokens=50)
    assert len(parts) > 1
    for p in parts:
        assert count_tokens(p) <= 50


def test_section_parent_blocks_created():
    para = "公司主营业务保持稳健发展，收入结构与去年基本一致，毛利率同比小幅提升。"
    text = "\n".join(["一、概览"] + [para] * 60)  # ≈ 1300 token → 多个叶子 + 1 个章节父块
    layout = LayoutResult(pages=[ParsedPage(page_no=1, text=text)])
    chunks = build_chunks("doc1", layout, build_section_tree(layout))
    parents = [c for c in chunks if c.chunk_type == "section"]
    assert parents  # 章节级父块存在
    assert chunks[0].chunk_type == "section"  # 父块顺序在前
    # 父块聚合了叶子内容（token 量级显著大于单叶子）
    assert parents[0].token_count >= 500
    # 每个叶子 parent_id 都指向存在的父块
    parent_ids = {p.id for p in parents}
    for leaf in [c for c in chunks if c.chunk_type != "section"]:
        assert leaf.parent_id in parent_ids


def test_body_adjacent_chunks_10pct_overlap():
    para = "报告期内公司实现营业收入86.5亿元，同比增长18.2%，毛利率29.6%，净利率14.3%，资产负债率58.9%。"
    text = "\n".join(["二、经营分析"] + [para] * 30)  # ≈ 1900 token → 多个相邻叶子
    layout = LayoutResult(pages=[ParsedPage(page_no=1, text=text)])
    chunks = build_chunks("doc1", layout, build_section_tree(layout))
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    assert len(text_chunks) >= 3
    # 相邻块保留重叠：下一块以本块尾部字符开头
    assert text_chunks[0].content[-10:] in text_chunks[1].content
    assert text_chunks[1].content[-10:] in text_chunks[2].content


def test_oversized_table_summary_and_pagination():
    headers = ["科目"] + [f"Q{i}" for i in range(10)]
    rows = [[f"科目{i}"] + [str(i * 10 + j) for j in range(10)] for i in range(300)]
    layout = LayoutResult(
        pages=[
            ParsedPage(
                page_no=1,
                text="财务明细",
                tables=[ParsedTable(page=1, headers=headers, rows=rows)],
            )
        ]
    )
    chunks = build_chunks("doc1", layout, build_section_tree(layout))
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) >= 2  # 摘要 + 分页
    # 摘要块含表头 + 规模统计
    assert any("本表共 300 行" in c.content for c in tables)
    # 分页明细完整（末行数据保留）
    joined = "\n".join(c.content for c in tables)
    assert "科目299" in joined


def test_table_pages_repeat_header_with_metadata():
    headers = ["科目"] + [f"Q{i}" for i in range(10)]
    rows = [[f"科目{i}"] + [str(i * 10 + j) for j in range(10)] for i in range(300)]
    layout = LayoutResult(
        pages=[
            ParsedPage(
                page_no=1,
                text="财务明细",
                tables=[ParsedTable(page=1, headers=headers, rows=rows)],
            )
        ]
    )
    chunks = build_chunks("doc1", layout, build_section_tree(layout))
    tables = [c for c in chunks if c.chunk_type == "table"]
    # 分页块各自携带表头 + 规模统计（独立可读）
    pages = [c for c in tables if "以下为分页明细" in c.content]
    assert len(pages) >= 2
    for c in pages:
        assert c.content.startswith("科目 | Q0")
    # 摘要块与分页块均带结构化元数据
    for c in tables:
        assert c.table_headers == headers
        assert c.n_rows == 300
        assert c.n_cols == 11
    # 加表头后分页块不超 table_max_tokens（2048）
    for c in tables:
        assert c.token_count <= 2048


def test_small_table_metadata_and_serialization():
    layout = LayoutResult(
        pages=[
            ParsedPage(
                page_no=1,
                text="收入明细",
                tables=[ParsedTable(page=1, headers=["季度", "营收"], rows=[["Q1", "18.9"]])],
            )
        ]
    )
    chunks = build_chunks("doc1", layout, build_section_tree(layout))
    table_chunk = next(c for c in chunks if c.chunk_type == "table")
    assert table_chunk.table_headers == ["季度", "营收"]
    assert table_chunk.n_rows == 1
    assert table_chunk.n_cols == 2
    # to_dict/from_dict 往返保留元数据
    restored = Chunk.from_dict(table_chunk.to_dict())
    assert restored.table_headers == ["季度", "营收"]
    assert restored.n_rows == 1
    assert restored.n_cols == 2
    # 文本块无表格元数据
    text_chunk = next(c for c in chunks if c.chunk_type == "text")
    assert text_chunk.table_headers is None
    assert text_chunk.n_rows is None
