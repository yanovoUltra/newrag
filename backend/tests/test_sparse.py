"""稀疏向量（jieba 分词 BM25 风格）单测。"""

from app.embed.sparse import sparse_embed, sparse_embed_one


def test_sparse_deterministic():
    a = sparse_embed_one("2024年公司营业收入为32.5亿元，同比增长15.2%")
    b = sparse_embed_one("2024年公司营业收入为32.5亿元，同比增长15.2%")
    assert a == b
    assert len(a["indices"]) == len(a["values"]) > 0
    assert all(isinstance(i, int) for i in a["indices"])
    assert all(v > 0 for v in a["values"])


def test_sparse_shared_tokens():
    doc_a = sparse_embed_one("营业收入增长")
    doc_b = sparse_embed_one("营业收入下降")
    set_a, set_b = set(doc_a["indices"]), set(doc_b["indices"])
    assert set_a & set_b  # 共享"营业收入"
    assert set_a ^ set_b  # 各自特有词


def test_sparse_stopwords_removed():
    out = sparse_embed_one("的是和在公司")
    assert out["indices"] == []


def test_sparse_empty():
    out = sparse_embed_one("")
    assert out == {"indices": [], "values": []}
    assert sparse_embed([]) == []
