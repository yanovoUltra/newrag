"""文档侧 IDF 重写 + 结构块检测单测。"""

from __future__ import annotations

import json

import pytest

from app.retrieval import idf


@pytest.fixture(autouse=True)
def _isolate_idf(tmp_path, monkeypatch):
    """隔离 idf 路径到临时目录，避免污染真实 data/idf.json。"""
    from app.core.config import get_settings

    monkeypatch.setitem(get_settings().__dict__, "sparse_idf_path", str(tmp_path / "idf.json"))
    monkeypatch.setitem(get_settings().__dict__, "sparse_idf_enabled", True)
    idf.reset_cache()
    yield
    idf.reset_cache()


def test_structural_detects_pure_financial_header():
    from app.parsers.structural import is_structural_chunk

    assert is_structural_chunk("（除另有标明外，所有金额均以人民币百万元列示）")
    assert is_structural_chunk(
        "上海浦东发展银行股份有限公司 截至2024年12月31日止年度财务报表 本集团 本行"
    )
    assert is_structural_chunk("单位：人民币百万元")


def test_structural_does_not_flag_data_tables():
    from app.parsers.structural import is_structural_chunk

    long_table = "营业收入 1,707,480 1,707,348 1,645,000 \n" * 20
    assert not is_structural_chunk(long_table)
    assert not is_structural_chunk("2024年公司实现营业收入人民币86.5亿元，同比增长18.2%。")
    assert not is_structural_chunk("")


def test_apply_idf_rewrites_weights():
    idf.idf_path().parent.mkdir(parents=True, exist_ok=True)
    idf.idf_path().write_text(json.dumps({"100": 2.0, "200": 0.5, "_idf": {"docs": 10}}), encoding="utf-8")
    idf.reset_cache()

    sparse = {"indices": [100, 200, 300], "values": [1.0, 1.0, 1.0]}
    out = idf.apply_idf(sparse)
    assert out["indices"] == [100, 300, 200]
    assert out["values"] == [2.0, 1.0, 0.5]


def test_apply_idf_disabled():
    from app.core.config import get_settings

    sparse = {"indices": [1, 2], "values": [0.5, 0.8]}
    get_settings().__dict__["sparse_idf_enabled"] = False
    assert idf.apply_idf(sparse) == sparse


def test_apply_idf_none_when_no_idf():
    assert idf.apply_idf({"indices": [1, 2], "values": [0.5, 0.8]}) == {"indices": [1, 2], "values": [0.5, 0.8]}
    assert idf.apply_idf(None) is None


def test_apply_structural_demote():
    from app.store import qdrant as qdrant_store

    hits = [
        {"chunk_id": "a", "is_structural": False, "score": 0.9},
        {"chunk_id": "b", "is_structural": True, "score": 0.8},
        {"chunk_id": "c", "is_structural": False, "score": 0.7},
    ]
    assert [h["chunk_id"] for h in qdrant_store._apply_structural(hits)] == ["a", "c", "b"]