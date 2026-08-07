"""BM25 风格稀疏向量：jieba 中文分词 → 词频权重，token 经稳定哈希映射为 Qdrant 稀疏索引。

配合 Qdrant sparse vector 的 IDF modifier，实现全局 BM25/TF-IDF 加权；
哈希映射保证入库与查询两端的索引空间一致（跨进程稳定）。
"""

from __future__ import annotations

import zlib
from collections import Counter

import jieba

# 少量高频停用词（中英文），避免噪声 token 抬高稀疏得分
_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "对", "为", "从", "于", "其", "之",
    "我们", "你们", "他们", "公司", "以及", "并且", "通过", "进行", "可以", "一个",
    "a", "an", "the", "of", "to", "in", "on", "and", "or", "for", "with",
}

_PUNCT = set("，。！？；：、（）《》【】“”‘’—…·,.:;!?'\"()[]{}\\|/`~@#$%^&*_+-=<>")


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for w in jieba.cut(text):
        w = w.strip().lower()
        if not w or w in _STOPWORDS or w in _PUNCT:
            continue
        if all(ch in _PUNCT for ch in w):
            continue
        tokens.append(w)
    return tokens


def _hash_token(token: str) -> int:
    # 稳定哈希：跨进程一致，冲突概率对数千词规模可忽略
    return zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF


def sparse_embed(texts: list[str]) -> list[dict]:
    """返回 Qdrant sparse 向量列表：[{"indices": [...], "values": [...]}]。"""
    out: list[dict] = []
    for text in texts:
        tf = Counter(_tokenize(text))
        indices: list[int] = []
        values: list[float] = []
        for token, cnt in tf.items():
            indices.append(_hash_token(token))
            values.append(float(cnt))
        out.append({"indices": indices, "values": values})
    return out


def sparse_embed_one(text: str) -> dict:
    return sparse_embed([text])[0]
