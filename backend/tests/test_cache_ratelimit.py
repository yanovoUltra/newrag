"""外部 API 缓解测试：嵌入结果缓存（output_type 隔离）、语义答案缓存、429 重试。

Redis 依赖型用例在 Redis 不可用时跳过（与 test_session_roundtrip 一致）。
"""

from __future__ import annotations

import pytest

from app.embed.embedder import ApiEmbedBackend
from app.store.redisx import get_redis


def _need_redis():
    if get_redis() is None:
        pytest.skip("Redis 不可用")


# ---- 嵌入结果缓存 ----

def test_embed_cache_roundtrip_and_output_type_isolation():
    _need_redis()
    from app.store.cache import embed_cache_get, embed_cache_set

    text, ttype = "营业收入1,707.48亿元。", "document"
    dense, sparse = [0.1] * 8, {"indices": [1, 2], "values": [0.9, 0.4]}
    embed_cache_set(text, ttype, "dense&sparse", dense, sparse)
    hit = embed_cache_get(text, ttype, "dense&sparse")
    assert hit is not None
    assert hit[0] == dense and hit[1] == sparse
    # 同文本不同 output_type 是不同 key（稀疏结果不能通用）
    assert embed_cache_get(text, ttype, "dense") is None


def test_embed_cache_norm_insensitive():
    _need_redis()
    from app.store.cache import embed_cache_get, embed_cache_set

    embed_cache_set("  营业收入1,707.48亿元  ", "document", "dense", [0.5] * 4, None)
    hit = embed_cache_get("营业收入1,707.48亿元", "document", "dense")
    assert hit is not None and hit[0] == [0.5] * 4


# ---- 语义答案缓存 ----

def test_answer_cache_lookup_threshold_and_dedup():
    _need_redis()
    from app.store.cache import answer_lookup, answer_store

    org, vis = "org_t", "public"
    q1 = [1.0, 0.0, 0.0]
    entry = {"q_norm": "浦发银行2024年营收是多少", "q_vec": q1, "events": [{"event": "done"}]}
    answer_store(org, vis, entry)
    # 高度相似问题命中（余弦 ≈1）
    hit = answer_lookup(org, vis, [0.999, 0.0, 0.001])
    assert hit is not None and hit["q_norm"] == entry["q_norm"]
    # 正交问题不命中（相似度 0 < 阈值 0.95）
    assert answer_lookup(org, vis, [0.0, 1.0, 0.0]) is None
    # 同 q_norm 写入去重替换
    answer_store(org, vis, {**entry, "events": [{"event": "done", "data": {"v": 2}}]})
    assert len(__load_bucket(org, vis)) == 1


def __load_bucket(org_id: str, user_visibility: str) -> list[dict]:
    import json

    r = get_redis()
    raw = r.get(f"answer_cache:{org_id}:{user_visibility}")
    return json.loads(raw) if raw else []


def test_answer_cache_qnorm_overlap_guard():
    """向量命中后二次校验：文本重合度过低视为可疑不命中（防御误命中）。"""
    _need_redis()
    import uuid

    from app.store.cache import answer_lookup, answer_store

    org = f"org_{uuid.uuid4().hex[:8]}"
    vis = "public"
    entry = {
        "q_norm": "浦发银行2024年营业收入是多少亿元",
        "q_vec": [1.0, 0.0, 0.0],
        "events": [{"event": "done"}],
    }
    answer_store(org, vis, entry)
    # 向量完全一致（相似度 1.0 >= 0.95）但文本重合度极低 → 不命中
    assert (
        answer_lookup(org, vis, [1.0, 0.0, 0.0], q_text="苹果公司2024财年净销售额是多少")
        is None
    )
    # 向量一致 + 文本高度重合（同义改写）→ 命中
    hit = answer_lookup(org, vis, [1.0, 0.0, 0.0], q_text="浦发银行2024年营业收入为多少亿元")
    assert hit is not None
    # 不传 q_text（旧调用方）→ 跳过重合度校验，按向量命中
    assert answer_lookup(org, vis, [1.0, 0.0, 0.0]) is not None


def test_fair_semaphore_fifo_order():
    """FairSemaphore：并发涌入时严格按到达顺序放行（asyncio.Semaphore 非公平会饿先到者）。"""
    import asyncio

    from app.generation.llm import FairSemaphore

    async def run():
        sem = FairSemaphore(2)
        order: list[int] = []

        async def worker(i):
            async with sem:
                order.append(i)
                await asyncio.sleep(0.01)

        await asyncio.gather(*(worker(i) for i in range(6)))
        return order

    assert asyncio.run(run()) == [0, 1, 2, 3, 4, 5]


# ---- 嵌入 API 限流 + 429 指数退避重试 ----

class _FakeResp:
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=None, response=self
            )

    def json(self):
        return self._json


def test_embed_post_batch_retries_on_429_then_succeeds():
    backend = ApiEmbedBackend("https://dashscope.aliyuncs.com", "sk-test", "m", 16)
    calls = {"n": 0}

    def fake_post(url, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            return _FakeResp(429)
        return _FakeResp(
            200,
            {"output": {"embeddings": [{"embedding": [0.1] * 8, "text_index": 0}]}},
        )

    backend._client.post = fake_post
    data = backend._post_batch(["x"], "dense", "query")
    assert calls["n"] == 3
    assert data["output"]["embeddings"][0]["embedding"] == [0.1] * 8


def test_embed_cached_call_avoids_repeat_api():
    _need_redis()
    import uuid

    backend = ApiEmbedBackend("https://dashscope.aliyuncs.com", "sk-test", "m", 16)
    calls = {"n": 0}
    prefix = uuid.uuid4().hex  # 每次运行唯一，避免历史缓存残留干扰

    def fake_call(texts, output_type, text_type):
        calls["n"] += 1
        dense = [[0.01 * i] * 8 for i in range(len(texts))]
        sparse = [{"indices": [i], "values": [1.0]} for i in range(len(texts))]
        return dense, sparse

    backend._call = fake_call
    a, b = f"{prefix}a", f"{prefix}b"
    # 首次：两个不同文本都未命中 → 一次批量调用
    dense1, sparse1 = backend._cached_call([a, b], "query", "dense&sparse")
    assert calls["n"] == 1
    # 再次：全部命中缓存 → 不再调用 API
    dense2, sparse2 = backend._cached_call([a, b], "query", "dense&sparse")
    assert calls["n"] == 1
    assert dense2 == dense1 and sparse2 == sparse1
