"""版式层测试：双栏 PDF 阅读顺序重组 + 单栏页不误判。"""

from __future__ import annotations

import os
import tempfile

import pytest

fitz = pytest.importorskip("fitz")


def _two_col_pdf(path: str) -> None:
    """生成合成双栏 PDF（英文，规避字体问题）：左栏 3 行 + 右栏 3 行。"""
    doc = fitz.open()
    p = doc.new_page()  # A4: 595 x 842
    # 左栏（x=72）
    p.insert_text((72, 80), "Left col line one", fontsize=11)
    p.insert_text((72, 100), "Left col line two", fontsize=11)
    p.insert_text((72, 120), "Left col line three", fontsize=11)
    # 右栏（x=340）
    p.insert_text((340, 80), "Right col line one", fontsize=11)
    p.insert_text((340, 100), "Right col line two", fontsize=11)
    p.insert_text((340, 120), "Right col line three", fontsize=11)
    doc.save(path)


def _single_col_pdf(path: str) -> None:
    """生成合成单栏 PDF（6 个窄块但都在中线左侧 → 不应触发双栏）。"""
    doc = fitz.open()
    p = doc.new_page()
    for i, y in enumerate(range(80, 220, 20)):
        p.insert_text((72, y), f"Single column line number {i}", fontsize=11)
    doc.save(path)


def test_two_column_page_reordered():
    """双栏页应输出"先左栏后右栏"（栏内按 y），而非左右交错。"""
    from app.parsers.layout import parse_pdf

    tmp = os.path.join(tempfile.mkdtemp(), "two_col.pdf")
    _two_col_pdf(tmp)
    layout = parse_pdf(__import__("pathlib").Path(tmp))
    text = layout.pages[0].text
    li = text.find("Left col line one")
    ri = text.find("Right col line one")
    assert li != -1 and ri != -1
    assert li < ri, "左栏内容应出现在右栏之前（当前左右交错）"
    # 左栏三行连续在前
    assert text.find("Left col line one") < text.find("Left col line two") < text.find("Left col line three")
    # 右栏三行连续在后
    assert text.find("Right col line one") < text.find("Right col line two") < text.find("Right col line three")


def test_single_column_not_flagged():
    """单栏页（全部窄块在中线左侧）不应被误判为双栏。"""
    from app.parsers.layout import parse_pdf

    tmp = os.path.join(tempfile.mkdtemp(), "single_col.pdf")
    _single_col_pdf(tmp)
    layout = parse_pdf(__import__("pathlib").Path(tmp))
    text = layout.pages[0].text
    # 单栏文本保持原有从上到下的顺序
    assert text.find("Single column line number 0") < text.find("Single column line number 5")


def test_two_column_helper_direct():
    """直接调用 _two_column_text：双栏页返回重组文本，单栏页返回 None。"""
    from app.parsers.layout import _two_column_text

    d1 = fitz.open()
    p = d1.new_page()
    for i in range(4):
        p.insert_text((60, 80 + i * 20), f"L{i}", fontsize=11)
        p.insert_text((340, 80 + i * 20), f"R{i}", fontsize=11)
    text = _two_column_text(p)
    assert text is not None
    # 左栏 4 行全部出现在右栏首行之前
    assert text.find("L3") < text.find("R0")

    d2 = fitz.open()
    p2 = d2.new_page()
    for i in range(4):
        p2.insert_text((60, 80 + i * 20), f"line{i}", fontsize=11)
    assert _two_column_text(p2) is None
