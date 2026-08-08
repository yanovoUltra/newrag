"""嵌入层单测：DashScope 原生响应解析、原生 base 推导、mock 后端协议。"""

import pytest

from app.embed.embedder import (
    EMBED_DIM,
    MockEmbeddingBackend,
    _native_base,
    _parse_native_response,
)


def _make_response(dense_dim=EMBED_DIM, with_sparse=True, n=2):
    items = []
    for i in range(n):
        item = {
            "embedding": [float(i) / 10.0] * dense_dim,
            "text_index": i,
        }
        if with_sparse:
            item["sparse_embedding"] = [
                {"index": 100 + i, "value": 0.8, "token": f"tok{i}a"},
                {"index": 200 + i, "value": 0.5, "token": f"tok{i}b"},
            ]
        items.append(item)
    # 打乱 text_index 顺序，验证按索引排序
    items.reverse()
    return {"output": {"embeddings": items}}


def test_native_base_maps_compatible_mode():
    assert _native_base("https://dashscope.aliyuncs.com/compatible-mode/v1") == (
        "https://dashscope.aliyuncs.com/api/v1"
    )


def test_native_base_keeps_native():
    url = "https://ws-x.cn-beijing.maas.aliyuncs.com/api/v1"
    assert _native_base(url) == url


def test_parse_dense_sparse():
    dense, sparse = _parse_native_response(_make_response(), "test-model")
    assert len(dense) == 2
    assert len(dense[0]) == EMBED_DIM
    # 乱序按 text_index 恢复
    assert dense[0][0] == 0.0
    assert dense[1][0] == 0.1
    assert sparse is not None and len(sparse) == 2
    assert sparse[0]["indices"] == [100, 200]
    assert sparse[0]["values"] == [0.8, 0.5]


def test_parse_sparse_missing_returns_none():
    dense, sparse = _parse_native_response(_make_response(with_sparse=False), "test-model")
    assert len(dense) == 2
    assert sparse is None


def test_parse_missing_dense_raises():
    data = {"output": {"embeddings": [{"text_index": 0, "sparse_embedding": []}]}}
    with pytest.raises(RuntimeError, match="缺少稠密向量"):
        _parse_native_response(data, "test-model")


def test_mock_with_sparse_returns_none_sparse():
    backend = MockEmbeddingBackend()
    dense, sparse = backend.embed_texts_with_sparse(["你好"], text_type="query")
    assert len(dense) == 1 and len(dense[0]) == EMBED_DIM
    assert sparse is None
    # text_type 不破坏确定性
    assert backend.embed_texts(["你好"]) == dense
