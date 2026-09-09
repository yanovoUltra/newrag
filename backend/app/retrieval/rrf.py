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


def rrf_fuse_route_aware(
    routes: list[list[dict]],
    k: int = 60,
    top_n: int | None = None,
) -> list[dict]:
    """Route-aware RRF：主分=1/(k+该 chunk 在所有 route 的最佳单路 rank)，tie-break=等权融合分。

    - score_primary = 1 / (k + min(rank over routes))  —— 治疗 R3(单路高排被等权稀释)
    - tie-breaker   = 等权融合分 Σ 1/(k+rank_route)      —— 多路都高排的 chunk 更 robust 优先
    两者都是语义信号（无 chunk_id 依赖），避免 G5 tie/order artifact。
    仅作 A/B 对照(A1)，不改生产等权默认。
    """
    best_rank: dict[str, int] = {}
    equal_score: dict[str, float] = {}
    ranking: dict[str, dict] = {}
    for route in routes:
        for rank, hit in enumerate(route, start=1):
            cid = hit["chunk_id"]
            if cid not in ranking:
                ranking[cid] = dict(hit)
            equal_score[cid] = equal_score.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in best_rank or rank < best_rank[cid]:
                best_rank[cid] = rank
    fused: list[dict] = []
    for cid, entry in ranking.items():
        e2 = dict(entry)
        e2["fused_score"] = 1.0 / (k + best_rank[cid])
        e2["fused_score_equal"] = equal_score.get(cid, 0.0)
        fused.append(e2)
    # best_rank 主序(小=前), 等权分次级(大=前); 两者皆语义, 无 chunk_id 依赖
    ranked = sorted(fused, key=lambda h: (-h["fused_score"], -h.get("fused_score_equal", 0.0)))
    if top_n is not None:
        return ranked[:top_n]
    return ranked
