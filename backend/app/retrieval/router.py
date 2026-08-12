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


# metric（指标型，关闭 IDF）/ entity（实体金额型，保留 IDF）/ general（通用）
QUERY_TYPES = ("metric", "entity", "general")

# 指标型关键词（比率/每股类，IDF 负收益 → 关闭 IDF），匹配优先级高于 entity
_METRIC_KEYWORDS = [
    "净资产收益率", "归母净利润", "扣非净利润", "净利润", "每股收益", "每股净资产",
    "每股经营现金流", "加权平均", "摊薄", "roaa", "roa", "roe", "eps",
    "毛利率", "净利率", "资产负债率", "权益乘数", "市盈率", "市净率", "总资产收益率",
    "净利", "每股",
]
# 实体金额型关键词（金额/经营主线，IDF 正收益 → 保留 IDF）
_ENTITY_KEYWORDS = [
    "营业收入", "营业总收入", "主营收入", "营收",
    "利润总额", "营业利润", "经营现金流", "经营活动产生的现金流量净额", "现金流",
    "货币资金", "总资产", "净资产", "所有者权益", "收入",
]

# HyDE 触发放宽触发词：综述/总结/分析类问题即使 LLM 未标复杂也生成假设文档
# （对抗性评测 §7.3 实证：D 类综述型 baseline Recall@8=0，HyDE 修复至 0.37）
# 2026-08-11 §15.5 P2：扩大到趋势/多跳/比较类（v9 非指标题趋势零分率 53.8%、
# 多跳 53.1%——"为什么/原因/趋势/对比"类问题跨块综合，单块检索易漏）
_SUMMARY_TRIGGERS = (
    "总结", "综述", "归纳", "概述", "概括", "梳理",
    "看法", "观点", "评价", "意见", "分析",
    "趋势", "走势", "变化", "变动", "发展",
    "为什么", "原因", "导致", "比较", "对比", "相比", "差异",
    "summar", "overview", "trend",
)


def _is_summary_query(question: str) -> bool:
    """启发式：问题是否属于综述/总结/分析型（多块综合，单块检索易漏）。"""
    q = (question or "").lower()
    return any(t in q for t in _SUMMARY_TRIGGERS)


@dataclass
class RoutePlan:
    intent: str = "factual"  # factual | abstract | multi_hop | table
    complexity: str = "simple"  # simple | complex
    query_type: str = "general"  # metric | entity | general
    rewritten_query: str | None = None
    sub_queries: list[str] = field(default_factory=list)
    sub_query_deps: dict[int, list[int]] = field(default_factory=dict)
    needs_hyde: bool = False

    @property
    def effective_queries(self) -> list[str]:
        """用于检索的有效查询集合：多跳子查询优先，其次改写查询，兜底原问题。

        注：DAG 拆解路径的依赖注入在 _search_plan 内按 sub_query_deps 执行，
        本属性仅返回扁平子查询文本（供简单多查询扩展场景兼容使用）。
        """
        if self.sub_queries:
            return self.sub_queries
        if self.rewritten_query:
            return [self.rewritten_query]
        return []


def classify_query_type_keyword(question: str) -> str:
    """关键词启发式分类：metric 优先（指标型），其次 entity（实体金额型），否则 general。

    用于 IDF 动态开关的查询类别判定；与 LLM 路由结果做"双重"合并（见 route_query）。
    """
    q = (question or "").lower()
    for kw in _METRIC_KEYWORDS:
        if kw in q:
            return "metric"
    for kw in _ENTITY_KEYWORDS:
        if kw in q:
            return "entity"
    return "general"


def _merge_query_type(plan_type: str, keyword_type: str) -> str:
    """双重合并：关键词给出明确 metric/entity 信号时优先生效；否则回退 LLM 结果。"""
    if keyword_type in ("metric", "entity"):
        return keyword_type
    return plan_type if plan_type in ("metric", "entity") else "general"


def _parse_sub_queries(raw) -> tuple[list[str], dict[int, list[int]]]:
    """解析 LLM 输出的子查询（新结构 [{step, question, dependency}] 与旧结构 [str] 兼容）。

    返回 (子查询列表, 依赖边 {index: [依赖子查询下标, ...]})，先过滤空项再截前 5 个。
    dependency 为 null/缺省表示无依赖；下标引用 sub_queries 数组中的位置（0 起）。
    """
    items: list[tuple[str, list[int] | None]] = []
    for item in raw or []:
        if isinstance(item, str):
            q = item.strip()
            if q:
                items.append((q, None))
            continue
        if not isinstance(item, dict):
            continue
        q = str(item.get("question") or "").strip()
        if not q:
            continue
        dep = item.get("dependency")
        if not isinstance(dep, list):
            dep = [dep]
        idxs = [
            int(d) for d in dep
            if d is not None and (
                isinstance(d, int) or (isinstance(d, str) and d.isdigit())
            )
        ]
        items.append((q, idxs or None))
    head = items[:5]
    subs = [q for q, _ in head]
    deps = {i: d for i, (_, d) in enumerate(head) if d}
    return subs, deps


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
    query_type = data.get("query_type", "general")
    if query_type not in QUERY_TYPES:
        query_type = "general"
    subs, sub_deps = _parse_sub_queries(data.get("sub_queries"))
    return RoutePlan(
        intent=intent,
        complexity=complexity,
        query_type=query_type,
        rewritten_query=str(data.get("rewritten_query") or "").strip() or None,
        sub_queries=subs,
        sub_query_deps=sub_deps,
        needs_hyde=bool(data.get("needs_hyde", False)),
    )


async def route_query(question: str) -> RoutePlan:
    """意图路由（JSON mode）。Mock LLM / 关闭路由时返回（默认计划 + 关键词类别）。"""
    settings = get_settings()
    keyword_type = classify_query_type_keyword(question)
    # HyDE 触发放宽：综述/总结/分析型问题强制开启（即使 LLM 标为 simple/factual）
    summary_trigger = _is_summary_query(question)
    if not settings.intent_routing_enabled:
        return RoutePlan(query_type=keyword_type, needs_hyde=summary_trigger)
    llm = get_llm()
    if llm.name == "mock":
        return RoutePlan(query_type=keyword_type, needs_hyde=summary_trigger)
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
        return RoutePlan(query_type=keyword_type, needs_hyde=summary_trigger)
    plan = _parse_plan(raw)
    # 双重合并：关键词信号优先，LLM 结果兜底
    plan.query_type = _merge_query_type(plan.query_type, keyword_type)
    plan.needs_hyde = plan.needs_hyde or summary_trigger
    return plan


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
