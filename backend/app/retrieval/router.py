"""LLM 检索编排：意图路由 + 查询重写 + 多跳拆分 + HyDE（阶段二，JSON mode）。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.generation.llm import get_llm
from app.generation.prompts import HYDE_PROMPT, ROUTER_PROMPT

logger = get_logger(__name__)


@dataclass
class RoutePlan:
    intent: str = "factual"  # factual | abstract | multi_hop | table
    complexity: str = "simple"  # simple | complex
    rewritten_query: str | None = None
    sub_queries: list[str] = field(default_factory=list)
    needs_hyde: bool = False

    @property
    def effective_queries(self) -> list[str]:
        """用于检索的有效查询集合：多跳子查询优先，其次改写查询，兜底原问题。"""
        if self.sub_queries:
            return self.sub_queries
        if self.rewritten_query:
            return [self.rewritten_query]
        return []


def _parse_plan(raw: str) -> RoutePlan:
    """容错解析 LLM JSON 输出（去代码围栏、补全缺省字段）。"""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 兜底：截取首个 { ... } 再解析
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            logger.warning("router json 解析失败，使用默认计划: %s", raw[:120])
            return RoutePlan()
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            logger.warning("router json 解析失败，使用默认计划: %s", raw[:120])
            return RoutePlan()

    intent = data.get("intent", "factual")
    if intent not in ("factual", "abstract", "multi_hop", "table"):
        intent = "factual"
    complexity = data.get("complexity", "simple")
    if complexity not in ("simple", "complex"):
        complexity = "simple"
    return RoutePlan(
        intent=intent,
        complexity=complexity,
        rewritten_query=str(data.get("rewritten_query") or "").strip() or None,
        sub_queries=[str(q).strip() for q in data.get("sub_queries", []) if str(q).strip()][:3],
        needs_hyde=bool(data.get("needs_hyde", False)),
    )


async def route_query(question: str) -> RoutePlan:
    """意图路由（JSON mode）。Mock LLM / 关闭路由时返回默认计划。"""
    settings = get_settings()
    if not settings.intent_routing_enabled:
        return RoutePlan()
    llm = get_llm()
    if llm.name == "mock":
        return RoutePlan()
    try:
        raw = await llm.chat(
            [
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": f"问题：{question}"},
            ],
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.warning("意图路由调用失败，使用默认计划: %s", e)
        return RoutePlan()
    return _parse_plan(raw)


async def generate_hypothetical_document(question: str) -> str | None:
    """HyDE：为抽象推理型问题生成假设文档（仅当 hyde_enabled）。"""
    settings = get_settings()
    if not settings.hyde_enabled:
        return None
    llm = get_llm()
    if llm.name == "mock":
        return None
    try:
        return await llm.chat(
            [
                {"role": "system", "content": HYDE_PROMPT.format(question=question)},
            ]
        )
    except Exception as e:
        logger.warning("HyDE 生成失败，跳过: %s", e)
        return None
