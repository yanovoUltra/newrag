"""问答编排：意图路由 → 混合检索（多路召回+RRF+rerank，含权限）→ 知识块组装 → 多模型流式生成。

支持：多轮会话历史注入（Redis，可选）、置信度门控、接地校验（数字回查知识块）。
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.embed.embedder import get_embedder
from app.generation.grounding import ground_answer
from app.generation.llm import get_llm, get_llm_light
from app.generation.prompts import build_answer_messages
from app.retrieval.reranker import get_reranker
from app.retrieval.router import RoutePlan, generate_hypothetical_document, route_query
from app.retrieval.search import hybrid_search
from app.store.cache import answer_lookup
from app.store.session import append_turn, get_history

logger = get_logger(__name__)

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _extract_year(text: str) -> int | None:
    """从问题中提取显式年份（如"2024年营收"→2024）；无显式年份返回 None（不过滤）。"""
    m = _YEAR_RE.search(text or "")
    return int(m.group(0)) if m else None


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
    fiscal_year: int | None = None,
) -> list[dict]:
    """按路由计划执行混合检索；复杂/抽象/多跳问题加深召回；可指定年份过滤。"""
    settings = get_settings()
    embedder = get_embedder()
    queries = plan.effective_queries or [question]
    all_hits: list[dict] = []
    hyde_doc: str | None = None
    # 复杂问题检索加深（放大各路 top_k 与 rerank 候选）
    depth = (
        settings.retrieval_depth_scale
        if plan.complexity == "complex" or plan.intent in ("abstract", "multi_hop")
        else 1.0
    )
    for q in queries:
        if plan.needs_hyde and hyde_doc is None:
            hyde_doc = await generate_hypothetical_document(question)
        # 查询侧一次取稠密+模型原生稀疏（text_type=query）。
        # 阻塞调用（嵌入 API / Qdrant / rerank）统一丢进线程池，避免占用事件循环
        # （否则单 worker 下并发问答会在检索段被串行化）。
        dense_list, sparse_list = await asyncio.to_thread(
            embedder.embed_texts_with_sparse, [hyde_doc or q], text_type="query"
        )
        hits = await asyncio.to_thread(
            hybrid_search,
            query_vector=dense_list[0],
            org_id=org_id,
            user_visibility=user_visibility,
            query_text=q,
            top_k=top_k,
            query_sparse=sparse_list[0] if sparse_list else None,
            recall_depth=depth,
            fiscal_year=fiscal_year,
        )
        all_hits.extend(hits)
    return _merge_blocks(all_hits, top_k)


def _best_score(blocks: list[dict]) -> float | None:
    """top 命中主分数：rerank 优先，其次 RRF 融合分。"""
    if not blocks:
        return None
    r = blocks[0]
    return r.get("rerank_score") if r.get("rerank_score") is not None else r.get("fused_score")


async def stream_answer(
    question: str,
    org_id: str,
    user_visibility: str,
    top_k: int | None = None,
    session_id: str = "",
) -> AsyncIterator[dict[str, Any]]:
    """按 SSE 事件协议产出：meta / citation / (warning) / token / grounding / done。

    外部 API 缓解：无会话的单轮问答走语义答案缓存——问题向量相似度命中则整份回放
    （省掉 rerank + LLM 生成两段最贵调用）；未命中则走完整流程并写入缓存。
    """
    settings = get_settings()
    k = top_k or settings.retrieval_top_k

    # 0. 语义答案缓存查询（仅无 session_id 的单轮问答；Redis 不可用自动未命中）
    cache_entry: dict | None = None
    q_vec: list[float] | None = None
    if settings.semantic_cache_enabled and not session_id:
        try:
            embedder = get_embedder()
            q_list, _ = await asyncio.to_thread(
                embedder.embed_texts_with_sparse, [question], text_type="query"
            )
            q_vec = q_list[0]
            cache_entry = answer_lookup(org_id, user_visibility, q_vec, q_text=question)
        except Exception as e:
            logger.warning("答案缓存查询失败，走完整流程: %s", e)
            cache_entry = None
    if cache_entry is not None:
        logger.info("语义答案缓存命中: org=%s q=%.40s", org_id, question)
        for ev in cache_entry.get("events", []):
            yield ev
        return

    events: list[dict[str, Any]] = []

    def _emit(ev: dict[str, Any]) -> dict[str, Any]:
        events.append(ev)
        return ev

    # 0.5 Guard 层（Agent 防线）：Prompt 注入 → 拒绝回答；循环 → 提示（均不阻断异常流程）
    if settings.guard_enabled:
        from app.generation.guard import check_conversation_loop, check_prompt_injection

        inj = check_prompt_injection(question)
        if inj.flagged:
            logger.warning("检测到 prompt 注入: %s", question[:60])
            yield _emit({
                "event": "meta",
                "data": {"org_id": org_id, "top_k": 0, "intent": "guard", "complexity": "simple",
                         "needs_hyde": False, "year": None, "year_fell_back": False,
                         "guarded": "injection"},
            })
            yield _emit({"event": "warning", "data": {"message": "检测到提示注入（指令覆盖/越权诱导），已拒绝处理该请求。"}})
            yield _emit({"event": "token", "data": {"delta": "抱歉，该请求包含提示注入特征，已拒绝处理。"}})
            yield _emit({"event": "done", "data": {"usage": {}}})
            return
        loop = check_conversation_loop(get_history(session_id), question)
        if loop:
            yield _emit({"event": "warning", "data": {"message": "检测到重复提问，本轮已回答。如需不同角度，请换一种问法。"}})

    # 1. 意图路由（阶段二；Mock/关闭时返回默认计划）
    plan = await route_query(question)

    # 2. 混合检索（陈旧文档防护：问题含显式年份 → 按年份过滤；无该年份文档时回退全量）
    year = _extract_year(question)
    blocks = await _search_plan(plan, question, org_id, user_visibility, k, fiscal_year=year)
    year_fell_back = False
    if year and not blocks:
        blocks = await _search_plan(plan, question, org_id, user_visibility, k, fiscal_year=None)
        year_fell_back = True

    yield _emit({
        "event": "meta",
        "data": {
            "org_id": org_id,
            "top_k": len(blocks),
            "intent": plan.intent,
            "complexity": plan.complexity,
            "needs_hyde": plan.needs_hyde,
            "year": year,
            "year_fell_back": year_fell_back,
        },
    })

    for b in blocks:
        yield _emit({
            "event": "citation",
            "data": {
                "doc_id": b.get("doc_id"),
                "doc_name": b.get("doc_name"),
                "section_path": b.get("section_path"),
                "page": b.get("page"),
                "chunk_type": b.get("chunk_type"),
                "score": round(b.get("rerank_score") or b.get("fused_score") or b.get("score") or 0.0, 4),
            },
        })

    if not blocks:
        yield _emit({"event": "token", "data": {"delta": "文档信息不足：未检索到相关知识块。"}})
        yield _emit({"event": "done", "data": {"usage": {}}})
        return

    # 3. 置信度门控（仅 rerank 模式）：top 命中分低于阈值 → 低置信提示
    best = _best_score(blocks)
    low_conf = (
        best is not None
        and get_reranker().name != "none"
        and settings.retrieval_min_score > 0
        and best < settings.retrieval_min_score
    )
    if low_conf:
        yield _emit({
            "event": "warning",
            "data": {"message": f"检索置信度较低（最高 {best:.3f}），回答可能不准确，请核对引用来源。"},
        })

    # 4. 多轮历史注入 + 多模型路由：简单问题走轻量模型（未配置则全部走主模型）
    history = get_history(session_id)
    messages = build_answer_messages(question, blocks, history=history)
    llm = get_llm_light() if plan.complexity == "simple" else None
    if llm is None:
        llm = get_llm()
    usage: dict[str, Any] = {}
    answer_text = ""
    had_error = False
    try:
        async for delta in llm.stream_chat(messages):
            answer_text += delta
            yield _emit({"event": "token", "data": {"delta": delta}})
    except Exception as e:
        had_error = True
        logger.exception("llm stream failed")
        yield _emit({"event": "error", "data": {"message": f"生成失败: {e}"}})

    # 5. 接地校验（数字/引用回查知识块）+ 会话记忆
    if answer_text:
        yield _emit({"event": "grounding", "data": ground_answer(answer_text, blocks)})
    append_turn(session_id, question, answer_text)
    yield _emit({"event": "done", "data": {"usage": usage}})

    # 5.5 问答记录持久化（离线复评 / RAGAS 数据来源；失败不影响回答）
    if settings.qa_record_enabled and answer_text:
        try:
            await asyncio.to_thread(
                _save_chat_record,
                question, answer_text, session_id, org_id, user_visibility,
                [b.get("doc_id") for b in blocks], plan.intent, usage,
            )
        except Exception as e:
            logger.warning("问答记录落库失败（忽略）: %s", e)

    # 6. 写入语义答案缓存（仅无会话 + 有完整回答 + 问题向量可用）
    if settings.semantic_cache_enabled and not session_id and q_vec and answer_text and not had_error:
        from app.store.cache import answer_store

        answer_store(
            org_id,
            user_visibility,
            {"q_norm": " ".join(question.split()).lower(), "q_vec": q_vec, "events": events},
        )


def _save_chat_record(
    question: str,
    answer: str,
    session_id: str,
    org_id: str,
    visibility: str,
    citation_doc_ids: list,
    intent: str,
    usage: dict,
) -> None:
    from app.store.registry import create_chat_record

    create_chat_record(
        question=question,
        answer=answer,
        session_id=session_id,
        org_id=org_id,
        visibility=visibility,
        citations=citation_doc_ids,
        intent=intent,
        usage=usage,
    )
