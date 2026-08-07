"""问答编排：意图路由 → 混合检索（多路召回+RRF+rerank，含权限）→ 知识块组装 → 多模型流式生成。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.core.logging import get_logger
from app.embed.embedder import get_embedder
from app.generation.llm import get_llm, get_llm_light
from app.generation.prompts import build_answer_messages
from app.retrieval.router import RoutePlan, generate_hypothetical_document, route_query
from app.retrieval.search import hybrid_search

logger = get_logger(__name__)


def _merge_blocks(results: list[dict], top_k: int) -> list[dict]:
    """多查询结果去重合并：同一 chunk 保留分数最高的一条，按分数降序取 top_k。"""
    best: dict[str, dict] = {}
    for r in results:
        cid = r["chunk_id"]
        if cid not in best:
            best[cid] = r
            continue
        score_key = "rerank_score" if best[cid].get("rerank_score") is not None else "fused_score"
        cur = r.get(score_key, 0.0)
        prev = best[cid].get(score_key, 0.0)
        if cur > prev:
            best[cid] = r
    ranked = sorted(best.values(), key=lambda h: h.get("rerank_score") or h.get("fused_score") or 0.0, reverse=True)
    return ranked[:top_k]


async def _search_plan(
    plan: RoutePlan,
    question: str,
    org_id: str,
    user_visibility: str,
    top_k: int,
) -> list[dict]:
    """按路由计划执行混合检索。"""
    embedder = get_embedder()
    queries = plan.effective_queries or [question]
    all_hits: list[dict] = []
    hyde_doc: str | None = None
    for q in queries:
        if plan.needs_hyde and hyde_doc is None:
            hyde_doc = await generate_hypothetical_document(question)
        dense_vec = embedder.embed_texts([hyde_doc or q])[0]
        hits = hybrid_search(
            query_vector=dense_vec,
            org_id=org_id,
            user_visibility=user_visibility,
            query_text=q,
            top_k=top_k,
        )
        all_hits.extend(hits)
    return _merge_blocks(all_hits, top_k)


async def stream_answer(
    question: str,
    org_id: str,
    user_visibility: str,
    top_k: int | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """按 SSE 事件协议产出：meta / citation / token / done。"""
    from app.core.config import get_settings

    settings = get_settings()
    k = top_k or settings.retrieval_top_k

    # 1. 意图路由（阶段二；Mock/关闭时返回默认计划）
    plan = await route_query(question)

    # 2. 混合检索（路由内同步完成；后续阶段移入线程池）
    blocks = await _search_plan(plan, question, org_id, user_visibility, k)

    yield {
        "event": "meta",
        "data": {
            "org_id": org_id,
            "top_k": len(blocks),
            "intent": plan.intent,
            "complexity": plan.complexity,
            "needs_hyde": plan.needs_hyde,
        },
    }

    for b in blocks:
        yield {
            "event": "citation",
            "data": {
                "doc_id": b.get("doc_id"),
                "doc_name": b.get("doc_name"),
                "section_path": b.get("section_path"),
                "page": b.get("page"),
                "chunk_type": b.get("chunk_type"),
                "score": round(b.get("rerank_score") or b.get("fused_score") or b.get("score") or 0.0, 4),
            },
        }

    if not blocks:
        yield {"event": "token", "data": {"delta": "文档信息不足：未检索到相关知识块。"}}
        yield {"event": "done", "data": {"usage": {}}}
        return

    # 3. 多模型路由：简单问题走轻量模型（未配置则全部走主模型）
    messages = build_answer_messages(question, blocks)
    llm = get_llm_light() if plan.complexity == "simple" else None
    if llm is None:
        llm = get_llm()
    usage: dict[str, Any] = {}
    try:
        async for delta in llm.stream_chat(messages):
            yield {"event": "token", "data": {"delta": delta}}
    except Exception as e:
        logger.exception("llm stream failed")
        yield {"event": "error", "data": {"message": f"生成失败: {e}"}}
    yield {"event": "done", "data": {"usage": usage}}
