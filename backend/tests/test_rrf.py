"""RRF 融合单测。"""

from app.retrieval.rrf import rrf_fuse


def _hit(cid: str, score: float) -> dict:
    return {"chunk_id": cid, "content": cid, "score": score}


def test_rrf_merge_rank():
    route_a = [_hit("a", 0.9), _hit("b", 0.8), _hit("c", 0.7)]
    route_b = [_hit("b", 0.6), _hit("c", 0.5)]
    fused = rrf_fuse([route_a, route_b], k=60)
    # b: 双路第1/2名；c: 双路第2/3名；a: 仅单路第1名
    assert [h["chunk_id"] for h in fused[:3]] == ["b", "c", "a"]
    # b 在两路中出现，fused_score 最高
    by_id = {h["chunk_id"]: h for h in fused}
    assert by_id["b"]["fused_score"] > by_id["c"]["fused_score"] > by_id["a"]["fused_score"]


def test_rrf_dedupe_and_route_count():
    route_a = [_hit("x", 1.0), _hit("y", 0.9)]
    route_b = [_hit("x", 0.5)]
    fused = rrf_fuse([route_a, route_b])
    assert len(fused) == 2  # x 去重
    x = next(h for h in fused if h["chunk_id"] == "x")
    assert x["routes"] == 2
    assert x["score"] == 1.0  # 保留最高原始分


def test_rrf_top_n():
    route_a = [_hit(str(i), 1.0 - i * 0.01) for i in range(10)]
    fused = rrf_fuse([route_a], top_n=3)
    assert len(fused) == 3
