"""字段抽取 + IDF 动态开关 单测。"""

from __future__ import annotations

from app.fields.extract import extract_fields
from app.fields.metrics import (
    extract_metric_from_question,
    match_metric_aliases,
    parse_number,
    parse_year,
)
from app.parsers.base import LayoutResult, ParsedPage, ParsedTable


# ---------- metrics：数值/年份/目录 ----------

def test_parse_number_commas_and_percent():
    v, u = parse_number("1,234,567.89")
    assert v == 1234567.89
    assert u is None


def test_parse_number_unit():
    v, u = parse_number("300,065.66 万元")
    assert v == 300065.66
    assert u == "万元"


def test_parse_number_percent():
    v, u = parse_number("12.34%")
    assert v == 12.34
    assert u == "%"


def test_parse_number_negative_paren():
    v, _ = parse_number("(1,234)")
    assert v == -1234.0


def test_parse_number_empty_tokens():
    for tok in ["—", "-", "不适用", "", "N/A"]:
        v, _ = parse_number(tok)
        assert v is None, tok


def test_parse_year():
    assert parse_year("2024年营业收入") == 2024
    assert parse_year("FY2024") == 2024
    assert parse_year("无年份") is None


def test_match_metric_aliases():
    m = match_metric_aliases("归属于母公司股东的净利润")
    assert m and m["key"] == "net_profit"
    m2 = match_metric_aliases("营业收入")
    assert m2 and m2["key"] == "revenue"


def test_extract_metric_from_question():
    key, year = extract_metric_from_question("2024年公司每股收益是多少？")
    assert key == "eps"
    assert year == 2024
    key2, year2 = extract_metric_from_question("今年营收怎么样")
    assert key2 == "revenue"
    assert year2 is None


# ---------- extract：从表格抽取 ----------

def _layout_with_table() -> LayoutResult:
    table = ParsedTable(
        page=1,
        headers=["项目", "2024年", "2023年"],
        rows=[
            ["营业收入", "300,065.66万元", "280,000.00万元"],
            ["归属于母公司股东的净利润", "12,345.67万元", "10,000.00万元"],
            ["基本每股收益", "0.88元", "0.75元"],
            ["加权平均净资产收益率", "12.34%", "11.00%"],
        ],
    )
    page = ParsedPage(page_no=1, text="", tables=[table])
    return LayoutResult(pages=[page])


def test_extract_fields_from_table():
    layout = _layout_with_table()
    records = extract_fields("doc1", layout)
    by_key = {(r.metric, r.year): r for r in records}
    assert ("revenue", 2024) in by_key
    assert by_key[("revenue", 2024)].value == 300065.66
    assert by_key[("revenue", 2024)].unit == "万元"
    assert ("net_profit", 2023) in by_key
    assert by_key[("eps", 2024)].value == 0.88
    assert by_key[("roe", 2024)].value == 12.34


def test_extract_transposed_table():
    table = ParsedTable(
        page=1,
        headers=["年份", "营业收入", "净利润"],
        rows=[
            ["2024年", "300,065.66", "12,345.67"],
            ["2023年", "280,000.00", "10,000.00"],
        ],
    )
    layout = LayoutResult(pages=[ParsedPage(page_no=1, text="", tables=[table])])
    records = extract_fields("doc2", layout)
    by_key = {(r.metric, r.year): r for r in records}
    assert ("revenue", 2024) in by_key
    assert by_key[("revenue", 2024)].value == 300065.66
    assert ("net_profit", 2023) in by_key


def test_extract_dedup_and_dedupe():
    records = extract_fields("doc1", _layout_with_table())
    keys = [(r.metric, r.year) for r in records]
    assert len(keys) == len(set(keys))


# ---------- neutralizel IDF ----------

def test_neutralize_idf_roundtrip():
    from app.retrieval import idf as idf_mod

    idf_mod.reset_cache()
    # 注入 IDF 映射
    idf_mod._IDF_CACHE = {1: 4.0, 2: 9.0, 3: 1.0}
    sparse = {"indices": [1, 2, 3], "values": [8.0, 18.0, 5.0]}
    neutral = idf_mod.neutralize_idf(sparse)
    vals = dict(zip(neutral["indices"], neutral["values"]))
    assert vals[1] == 2.0  # 8 / 4
    assert vals[2] == 2.0  # 18 / 9
    assert vals[3] == 5.0  # 5 / 1（未加权 index 不变）
    idf_mod.reset_cache()