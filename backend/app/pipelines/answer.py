"""问答编排：向量化问题 → 检索（含权限）→ 构造知识块 → 流式生成。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.core.logging import get_logger
from app.embed.embedder import get_embedder
from app.generation.llm import get_llm
from app.generation.prompts import build_answer_messages
from app.retrieval.search import retrieve

logger = get_logger(__name__)


def search_context(
    question: str,
    org_id: str,
    user_visibility: str,
    top_k: int | None = None,
) -> list[dict]:
    """检索并返回知识块列表（含来源标注所需字段）。"""
    embedder = get_embedder()
    query_vector = embedder.embed_texts([question])[0]
    return retrieve(
        query_vector=query_vector,
        org_id=org_id,
        user_visibility=user_visibility,
        top_k=top_k,
    )


async def stream_answer(
    question: str,
    org_id: str,
    user_visibility: str,
    top_k: int | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """按 SSE 事件协议产出：meta / citation / token / done。"""
    # 阶段一：检索在事件流内同步完成（简单问题首 token 目标 <3s，后续优化）
    blocks = search_context(question, org_id, user_visibility, top_k)
    yield {"event": "meta", "data": {"org_id": org_id, "top_k": len(blocks)}}

    for b in blocks:
        yield {
            "event": "citation",
            "data": {
                "doc_id": b.get("doc_id"),
                "doc_name": b.get("doc_name"),
                "section_path": b.get("section_path"),
                "page": b.get("page"),
                "chunk_type": b.get("chunk_type"),
                "score": round(b.get("score", 0.0), 4),
            },
        }

    if not blocks:
        yield {"event": "token", "data": {"delta": "文档信息不足：未检索到相关知识块。"}}
        yield {"event": "done", "data": {"usage": {}}}
        return

    messages = build_answer_messages(question, blocks)
    llm = get_llm()
    usage: dict[str, Any] = {}
    try:
        async for delta in llm.stream_chat(messages):
            yield {"event": "token", "data": {"delta": delta}}
    except Exception as e:
        logger.exception("llm stream failed")
        yield {"event": "error", "data": {"message": f"生成失败: {e}"}}
    yield {"event": "done", "data": {"usage": usage}}
