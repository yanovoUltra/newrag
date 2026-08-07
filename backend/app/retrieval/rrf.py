"""RRF（Reciprocal Rank Fusion）：多路召回融合，k 默认 60。"""

from __future__ import annotations


def rrf_fuse(
    routes: list[list[dict]],
    k: int = 60,
    top_n: int | None = None,
) -> list[dict]:
    """将多路召回（每路按相关性降序）融合为单一排序列表。

    score = sum(1 / (k + rank))，按 chunk_id 去重；返回前 top_n 条。
    结果附带 fused_score 与每路的原始 score（取最高）。
    """
    fused: dict[str, dict] = {}
    for route in routes:
        for rank, hit in enumerate(route, start=1):
            cid = hit["chunk_id"]
            entry = fused.setdefault(cid, dict(hit))
            entry["fused_score"] = entry.get("fused_score", 0.0) + 1.0 / (k + rank)
            entry["score"] = max(entry.get("score", 0.0), hit.get("score", 0.0))
            entry.setdefault("routes", 0)
            entry["routes"] += 1
    ranked = sorted(fused.values(), key=lambda h: h["fused_score"], reverse=True)
    if top_n is not None:
        return ranked[:top_n]
    return ranked
