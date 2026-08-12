"""§18 召回端优化单测：数字+单位短语提取 + 章节父块检索路。"""

import pytest

from app.fields.phrases import extract_number_phrases
from app.retrieval.search import _extract_exact_phrases, hybrid_search


class _MockReranker:
    name = "mock"

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        return [0.5] * len(docs)


class _MockEmbedder:
    def embed_texts_with_sparse(self, texts, text_type="query"):
        n = len(texts)
        return (
            [[0.1] * 8] * n,
            [{"indices": [0], "values": [1.0]}] * n,
        )


@pytest.fixture
def mock_search_env(monkeypatch):
    """隔离外部依赖：Qdrant 检索/权限/rerank/嵌入/主体预过滤/父块上下文全部打桩。"""
    dense_calls: list[dict] = []
    sparse_calls: list[dict] = []

    def _fake_dense(**kw):
        dense_calls.append(kw)
        return []

    def _fake_sparse(**kw):
        sparse_calls.append(kw)
        return []

    monkeypatch.setattr("app.store.qdrant.search_dense", _fake_dense)
    monkeypatch.setattr("app.store.qdrant.search_sparse", _fake_sparse)
    monkeypatch.setattr("app.store.qdrant.visible_levels", lambda vis: [vis])
    monkeypatch.setattr("app.store.qdrant.fetch_payloads", lambda *a, **k: {})
    monkeypatch.setattr("app.retrieval.search._resolve_subject_doc_ids", lambda *a, **k: None)
    monkeypatch.setattr("app.retrieval.search.get_reranker", lambda: _MockReranker())
    monkeypatch.setattr("app.embed.embedder.get_embedder", lambda: _MockEmbedder())
    return {"dense": dense_calls, "sparse": sparse_calls}


def test_hybrid_search_parent_route_off(mock_search_env):
    hybrid_search(
        query_vector=[0.1] * 8, org_id="default", user_visibility="public",
        query_text="浦发银行2024年不良贷款率是多少？", query_sparse={"indices": [0], "values": [1.0]},
        use_parent_route=False,
    )
    assert not any(c.get("chunk_type") == "section" for c in mock_search_env["dense"])


def test_hybrid_search_parent_route_on(mock_search_env):
    hybrid_search(
        query_vector=[0.1] * 8, org_id="default", user_visibility="public",
        query_text="总结浦发银行管理层讨论与分析中的风险观点",
        query_sparse={"indices": [0], "values": [1.0]},
        use_parent_route=True,
    )
    assert any(c.get("chunk_type") == "section" for c in mock_search_env["dense"])


def test_hybrid_search_number_phrase_route(mock_search_env):
    """数字+单位短语应触发短语路稀疏检索（短语含数字时）。"""
    hybrid_search(
        query_vector=[0.1] * 8, org_id="default", user_visibility="public",
        query_text="浦发银行2024年不良贷款率是1.36%吗？",
        query_sparse={"indices": [0], "values": [1.0]},
        use_phrase_route=True,
    )
    # 短语路每短语一次稀疏检索：断言发生过多路稀疏调用（原查询 1 次 + 短语路 ≥1 次）
    assert len(mock_search_env["sparse"]) >= 2


# ---- 数字+单位短语提取 ----

def test_number_phrase_percent():
    assert extract_number_phrases("浦发银行2024年不良贷款率是1.36%吗？") == ["1.36%"]


def test_number_phrase_yi():
    assert extract_number_phrases("中芯国际2025年营收是673.2亿元吗？") == ["673.2亿元"]


def test_number_phrase_excludes_year():
    # 4 位年份（无单位）不提取；"2024年"中"年"不是单位
    assert extract_number_phrases("浦发银行2024年营业收入是多少？") == []


def test_number_phrase_excludes_plain_number():
    assert extract_number_phrases("请说明不良贷款率的变化情况") == []


def test_number_phrase_range_extracts_both():
    assert extract_number_phrases("毛利率在1.5%~2.0%区间") == ["1.5%", "2.0%"]


def test_number_phrase_dedup():
    assert extract_number_phrases("增长3倍，预计达到3倍以上") == ["3倍"]


def test_number_phrase_thousand_sep():
    assert extract_number_phrases("净利润12,345.6万元") == ["12,345.6万元"]


# ---- 精确短语（回归：合并数字短语不影响引号短语） ----

def test_exact_phrase_unchanged():
    assert _extract_exact_phrases('《内部审计制度》何时修订？') == ["内部审计制度"]


def test_exact_phrase_no_number_pollution():
    assert _extract_exact_phrases("不良贷款率是1.36%吗？") == []
