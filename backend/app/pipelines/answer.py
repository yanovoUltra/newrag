"""问答编排：意图路由 → 混合检索（多路召回+RRF+rerank，含权限）→ 知识块组装 → 多模型流式生成。

支持：多轮会话历史注入（Redis，可选）、置信度门控、接地校验（数字回查知识块）。
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger, get_trace_id, safe_ref
from app.core.otel import make_span
from app.embed.embedder import get_embedder
from app.fields.metrics import extract_metric_from_question
from app.fields.phrases import extract_number_phrases
from app.fields.subject import find_subject_company
from app.generation.grounding import ground_answer
from app.generation.llm import get_llm, get_llm_light
from app.generation.prompts import build_answer_messages
from app.retrieval.reranker import get_reranker
from app.retrieval.router import (
    RoutePlan,
    _is_summary_query,
    generate_hypothetical_document,
    route_query,
)
from app.retrieval.search import _extract_exact_phrases, hybrid_search
from app.store.cache import answer_lookup
from app.store.registry import list_field_companies, log_fallback
from app.store.session import append_turn, get_history

logger = get_logger(__name__)

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _extract_year(text: str) -> int | None:
    """从问题中提取显式年份（如"2024年营收"→2024）；无显式年份返回 None（不过滤）。"""
    m = _YEAR_RE.search(text or "")
    return int(m.group(0)) if m else None


def _query_metric_keys(text: str) -> set[str]:
    """Find independent explicit metrics, suppressing nested generic aliases."""
    from app.fields.metrics import METRIC_CATALOG

    text = text.casefold()
    matches = [
        (match.start(), match.end(), metric["key"])
        for metric in METRIC_CATALOG
        for alias in metric["aliases"] if alias
        for match in re.finditer(re.escape(alias.casefold()), text)
    ]
    return {key for start, end, key in matches if not any(
        other_start <= start and end <= other_end
        and other_end - other_start > end - start
        for other_start, other_end, _ in matches
    )}


def _direct_field_relay_eligible(question: str, org_id: str, visibility: str) -> bool:
    """Permit a direct fact despite an abstract route only with exact field proof.

    No classifier label, Gold value or company-specific exception is used.
    Ambiguity, missing provenance and analytical requests retain normal routing.
    """
    from app.fields.subject import find_mentioned_companies
    from app.store.registry import query_fields

    if not re.search(r"多少|几元|\bwhat\s+(?:is|was|were|are)\b|\bhow\s+much\b", question, re.I):
        return False
    if re.search(
        r"趋势|变化|增长|下降|增加|减少|原因|影响|风险|战略|治理|比较|对比|分析|为什么|如何|"
        r"是否|不是|不含|不包括|除外|计算|占比|差额|差值|分别|各自|哪些|"
        r"\b(?:trend|change|growth|increase|decrease|why|risk|compare|comparison|"
        r"difference|calculate|not|excluding|except|versus|ratio|percentage|each)\b",
        question, re.I,
    ):
        return False
    metrics = _query_metric_keys(question)
    years = {int(m.group()) for m in _YEAR_RE.finditer(question)}
    if len(metrics) != 1 or len(years) != 1:
        return False
    try:
        companies = list_field_companies(org_id, visibility)
        company = find_subject_company(question, companies)
        if company is None or len(find_mentioned_companies(question, companies)) > 1:
            return False
        year = next(iter(years))
        rows = query_fields(list(metrics), org_id, visibility, year=year, company=company, limit=2)
        return (len(rows) == 1 and rows[0].year == year
                and bool(rows[0].source_chunk_id))
    except Exception:
        return False


def _constraint_preserving_query(
    original: str, rewritten: str | None, org_id: str, user_visibility: str,
) -> str:
    """Reject a rewrite that changes explicit metric/year/company identities.

    This is a retrieval guard, not a new answer classifier. It reads no labels
    or values, never changes the route intent, and leaves subquery DAGs alone.
    Unknown company metadata fails closed to the original question.
    """
    if not rewritten or rewritten == original:
        return original
    from app.fields.subject import find_mentioned_companies

    if (_query_metric_keys(original) != _query_metric_keys(rewritten)
            or {m.group() for m in _YEAR_RE.finditer(original)}
            != {m.group() for m in _YEAR_RE.finditer(rewritten)}):
        return original
    try:
        companies = list_field_companies(org_id, user_visibility)
        old = set(find_mentioned_companies(original, companies))
        new = set(find_mentioned_companies(rewritten, companies))
        old_subject = find_subject_company(original, companies)
        new_subject = find_subject_company(rewritten, companies)
        if old != new or old_subject != new_subject:
            return original
    except Exception:
        return original
    return rewritten


_TREND_KEYWORDS = ("相比", "对比", "趋势", "变化", "演进", "逐年", "同期", "变动", "增长情况")
_YEAR_RANGE_RE = re.compile(r"(19|20)\d{2}\s*年?\s*(?:至|到|~|—|-)\s*(?:19|20)\d{2}")
_VERIFICATION_KEYWORDS = (
    "辨析",
    "核实",
    "传闻",
    "市场观点",
    "说法",
    "准确性",
    "是否属实",
    "verify",
    "claim",
    "rumor",
)


def _is_trend_query(text: str) -> bool:
    """趋势/跨年对比型问题：含比较词或"2023至2025年"式年份范围。

    §17.4 探针：趋势题被单一 fiscal_year 过滤截断（相关块跨年分布），
    CXMT 毛利率趋势/季度占比/Tesla 研发 3 题 相关块排名 >50→top5（过滤放宽后）。
    """
    if any(k in text for k in _TREND_KEYWORDS):
        return True
    return bool(_YEAR_RANGE_RE.search(text or ""))


def _is_verification_query(text: str) -> bool:
    """Whether a mentioned year may belong to the claim rather than the source report."""

    normalized = (text or "").casefold()
    return any(keyword in normalized for keyword in _VERIFICATION_KEYWORDS)


def _effective_fiscal_year(text: str, fiscal_year: int | None) -> int | None:
    """Keep strict years for facts, but relax cross-period and claim-verification queries."""

    if fiscal_year is not None and (
        _is_trend_query(text) or _is_verification_query(text)
    ):
        return None
    return fiscal_year


def _lookup_field_evidence(
    question: str, org_id: str, user_visibility: str, year: int | None
) -> list[dict]:
    """字段抽取检索：metric 类问题→定位主体公司，按 (指标, 年份, 公司) 查独立字段索引。

    主体公司无法确定（无公司名/多个公司）时返回空——避免把多公司混合值当"强证据"注入。
    显式年份无命中时回退该公司全年份。
    """
    from app.fields.metrics import extract_metric_from_question
    from app.fields.subject import find_subject_company
    from app.store.registry import get_document, list_field_companies, query_fields

    key, q_year = extract_metric_from_question(question)
    if key is None:
        return []
    company = find_subject_company(question, list_field_companies(org_id, user_visibility))
    if company is None:
        return []
    target_year = q_year or year
    rows = query_fields([key], org_id, user_visibility, year=target_year, company=company, limit=20)
    if not rows and target_year is not None:
        rows = query_fields([key], org_id, user_visibility, year=None, company=company, limit=20)
    evidence: list[dict] = []
    for r in rows:
        doc = get_document(r.doc_id)
        evidence.append(
            {
                "metric": r.metric,
                "metric_label": r.metric_label,
                "year": r.year,
                "value": r.value,
                "unit": r.unit,
                "raw": r.raw,
                "doc_id": r.doc_id,
                "doc_name": doc.filename if doc else r.doc_id,
                "company": r.company,
                "page": r.page,
                "section_path": r.section_path,
            }
        )
    return evidence


def _field_block(ev: dict) -> dict:
    """把字段证据组装成知识块（供 LLM 引用精确取值），来源标注与普通块一致。"""
    src = f"{ev['doc_name']}-{ev['section_path']}-第{ev['page']}页"
    return {
        "doc_id": ev["doc_id"],
        "doc_name": ev["doc_name"],
        "page": ev["page"],
        "section_path": ev["section_path"],
        "chunk_type": "field",
        "content": f"【字段】{ev['metric_label']} {ev['year']}年：{ev['raw']}（来源：{src}）",
        "parent_content": "",
    }


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
    ranked = sorted(
        best.values(),
        key=lambda h: (
            bool(h.get("is_relay")),
            h.get("rerank_score") or h.get("fused_score") or 0.0,
        ),
        reverse=True,
    )
    return ranked[:top_k]


# ---- DAG 子查询：依赖注入 + 拓扑执行 ----
# 中文所有格占位（替换为 "实体+的"，如"该银行的生息资产结构"→"建设银行的生息资产结构"）
_SUBQ_POSSESSIVE = ("该银行的", "该行的", "该公司的", "这家银行的", "此银行的")
# 普通占位（长模式在前，避免 "该银行" 被 "该行" 提前替换误伤）
_SUBQ_PLACEHOLDERS = (
    "该银行", "该行", "该公司", "这家银行", "此银行",
    "the bank's", "that bank's", "the company's", "that company's",
    "that bank", "the bank", "that company", "the company",
)


def _inject_dependency_entity(query: str, entity: str) -> str:
    """把前序子查询解析出的实体替换进依赖查询的占位词。

    例："该行2024年拨备覆盖率是多少？" + entity="农业银行" → "农业银行2024年拨备覆盖率是多少？"
        "该银行的生息资产结构如何？" + entity="建设银行" → "建设银行的生息资产结构如何？"
    """
    out = query
    for pat in _SUBQ_POSSESSIVE:
        if pat in out:
            out = out.replace(pat, f"{entity}的")
    for pat in _SUBQ_PLACEHOLDERS:
        if pat in out:
            out = out.replace(pat, entity)
    return out


def _topo_order(deps: dict[int, list[int]], n: int) -> list[int]:
    """依赖边拓扑排序（Kahn）；环/越界依赖兜底按原序追加（不阻塞执行）。"""
    from collections import deque

    indeg = [0] * n
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, ds in deps.items():
        if not (0 <= i < n):
            continue
        for d in ds:
            if 0 <= d < n and d != i:
                adj[d].append(i)
                indeg[i] += 1
    queue = deque(i for i in range(n) if indeg[i] == 0)
    order: list[int] = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    seen = set(order)
    order.extend(i for i in range(n) if i not in seen)
    return order


def _extract_answer_entity(
    hits: list[dict], query: str, companies: list[str]
) -> str | None:
    """从命中块/查询本身提取"答案实体"（供注入依赖查询）。

    优先级：命中块内容识别出的公司 > 查询自身主体公司。
    例："2024年哪家国有行不良率最低" 的 top 命中块内容含"农业银行" → 返回"农业银行"。
    """
    for h in hits:
        comp = find_subject_company(h.get("content") or "", companies)
        if comp:
            return comp
    return find_subject_company(query, companies)


def _expand_query_aliases(q: str) -> str:
    """指标别名扩召回：命中指标时把核心规范别名拼接进查询文本（稠密/稀疏/rerank 共用）。

    例："2025年归母净利润是多少" → "2025年归母净利润是多少 归属于母公司股东的净利润"
    ——同义词归一化，解决"表述不同但语义一致"（简称/全称/英文缩写）的字面漏召。
    已在 query 中的别名跳过；无指标命中返回原句。
    """
    from app.fields.metrics import METRIC_CATALOG

    key, _ = extract_metric_from_question(q)
    if not key:
        return q
    metric = next((m for m in METRIC_CATALOG if m["key"] == key), None)
    if not metric:
        return q
    ql = q.lower()
    extras = [a for a in metric["aliases"] if a and a.lower() not in ql][:2]
    return f"{q} {' '.join(extras)}" if extras else q


def _hyde_disallowed(q: str, org_id: str | None = None,
                     user_visibility: str | None = None) -> bool:
    """HyDE 场景分流：精确信号强的查询禁用 HyDE（假设文档会稀释原句约束）。

    禁用：带"数值+单位"（数值直查）、带引号/书名号（短语锚点）、纯指标直查
    （无分析词；relay 字段索引已精确覆盖）。保留：分析/趋势/综述等泛化需求题。
    纯指标直查判定用公司感知版综述检查（"上海浦东发展银行"里的"发展"不是意图词，
    否则此类题误开 HyDE——见 _summary_query_company_aware）。
    """
    if extract_number_phrases(q):
        return True
    if _extract_exact_phrases(q):
        return True
    key, _ = extract_metric_from_question(q)
    if key and not _summary_query_company_aware(q, org_id, user_visibility):
        return True
    return False


def _summary_query_company_aware(q: str, org_id: str | None,
                                 user_visibility: str | None) -> bool:
    """综述触发词判定（公司实体感知）：先剔除题中公司名再查触发词。

    背景：_SUMMARY_TRIGGERS 含"发展"等通用词，而"上海浦东发展银行"公司名也含
    "发展"——实体 span 不是意图词，命中的话纯指标题会被误判成综述型 → 关 table/
    relay + 误开 HyDE/父块/别名（诊断：157 指标题中 17 题浦发 relay 全失）。
    剔除后"2024年业绩发展趋势分析"仍保留"趋势"等真触发词，不误伤真综述题。
    """
    if not org_id or not user_visibility:
        return _is_summary_query(q)
    try:
        from app.store.registry import list_field_companies

        for c in list_field_companies(org_id, user_visibility) or []:
            if c and c in q:
                q = q.replace(c, "")
    except Exception:  # noqa: BLE001  registry 未初始化等场景退化为原判定
        pass
    return _is_summary_query(q)


def _is_narrative_query(q: str) -> bool:
    """叙述型事实题兜底（2026-08-11）：无指标命中且无综述触发词 → 答案位于叙述性
    文本块而非表格数值（如 "What was GE's property, plant and equipment in 1998?"、
    "营业网点有多少个"），table 路对其负贡献（§15.3）→ 视为分析类关 table/relay。

    即使 LLM 把此类题误判为 factual，也由本判定兜底（生产 _search_one 的
    analysis 判定覆盖 intent + summary 触发词 + 本函数三层）。
    """
    key, _ = extract_metric_from_question(q)
    return key is None and not _is_summary_query(q)


def _merge_grouped(per_group: dict[int, list[dict]], order: list[int], top_k: int) -> list[dict]:
    """基线分组合并：每个多跳子查询独立取 Top-N，再全局按分补齐。

    保证每个子查询都有代表证据（多跳/对比题信息均衡，避免单一子查询高相关结果
    占满全部上下文）；组内按 rerank/fused 分排序。
    """
    n = max(1, len(order))
    group_k = max(2, top_k // n)
    chosen: list[dict] = []
    seen: set[str] = set()

    for index in order:
        for hit in _merge_blocks(per_group.get(index, []), group_k):
            if hit["chunk_id"] in seen:
                continue
            seen.add(hit["chunk_id"])
            chosen.append(hit)
    if len(chosen) < top_k:
        all_sorted = sorted(
            (hit for group in per_group.values() for hit in group),
            key=lambda hit: hit.get("rerank_score")
            or hit.get("fused_score")
            or 0.0,
            reverse=True,
        )
        for hit in all_sorted:
            if len(chosen) >= top_k:
                break
            if hit["chunk_id"] in seen:
                continue
            seen.add(hit["chunk_id"])
            chosen.append(hit)
    return chosen[:top_k]


def _merge_coverage_groups(
    per_group: dict[int, list[dict]], order: list[int], top_k: int
) -> list[dict]:
    """方面实验专用：每组先保活一个，再轮转补齐并去除近重复。"""
    ranked_groups = {
        index: _merge_blocks(per_group.get(index, []), top_k) for index in order
    }
    chosen: list[dict] = []
    seen: set[str] = set()
    content_keys: set[str] = set()

    def add(hit: dict) -> bool:
        content_key = re.sub(r"[^\w]+", "", str(hit.get("content") or "").casefold())[:240]
        if hit["chunk_id"] in seen or (content_key and content_key in content_keys):
            return False
        seen.add(hit["chunk_id"])
        if content_key:
            content_keys.add(content_key)
        chosen.append(hit)
        return True

    # Coverage pass: every aspect/sub-query gets one slot before any group gets a second.
    for index in order[:top_k]:
        for hit in ranked_groups.get(index, []):
            if add(hit):
                break
    # Round-robin pass prevents the first group from consuming the whole budget.
    depth = 1
    while len(chosen) < top_k:
        progressed = False
        for index in order:
            group = ranked_groups.get(index, [])
            if depth >= len(group):
                continue
            progressed = add(group[depth]) or progressed
            if len(chosen) >= top_k:
                break
        if not progressed and all(depth >= len(ranked_groups.get(index, [])) - 1 for index in order):
            break
        depth += 1
    if len(chosen) < top_k:
        all_sorted = sorted(
            (h for g in per_group.values() for h in g),
            key=lambda h: h.get("rerank_score") or h.get("fused_score") or 0.0,
            reverse=True,
        )
        for h in all_sorted:
            if len(chosen) >= top_k:
                break
            add(h)
    return chosen[:top_k]


# 对比题实体均衡保活（八）：对比关键词 + ≥2 核心实体 → 按实体分组各取 Top-M。
# "和/与/跟"作对比词由"≥2 实体"闸门兜底（"营收和净利润"仅 1 实体不误触发）。
_COMPARISON_KEYWORDS = ("对比", "比较", "分别", "差距", "相比", "差别", "差异",
                        "和", "与", "跟", "vs", "versus")
_ENTITY_ABBREV = {
    "工行": "工商银行", "建行": "建设银行", "农行": "农业银行", "中行": "中国银行",
    "招行": "招商银行", "浦发": "浦东发展银行", "交行": "交通银行",
    "茅台": "贵州茅台", "中芯": "中芯国际", "宁德": "宁德时代", "海康": "海康威视",
    "汇川": "汇川技术", "比亚迪": "比亚迪",
}
_COMPANY_PREFIX = ("中国", "上海", "深圳", "宜宾", "杭州", "北京市", "上海市")
_COMPANY_SUFFIX = ("股份有限公司", "有限公司", "股份", "集团")


def _company_core_keys(name: str) -> set[str]:
    """公司名核心 key 集：全名 + 去前后缀核心名（"中国工商银行"→"工商银行"，繁体归一）。"""
    from app.fields.subject import _normalize_name

    s = _normalize_name(name or "")
    keys = {s}
    core = s
    for pre in _COMPANY_PREFIX:
        if core.startswith(pre):
            core = core[len(pre):]
            break
    for suf in _COMPANY_SUFFIX:
        if core.endswith(suf):
            core = core[: -len(suf)]
            break
    keys.add(core)
    return {k for k in keys if k}


def _detect_comparison_entities(q: str, companies: list[str]) -> list[str]:
    """对比意图 + ≥2 核心实体检测：返回实体公司名（简称/全名均可）；否则 []。"""
    ql = (q or "").lower()
    if not any(k in ql for k in _COMPARISON_KEYWORDS):
        return []
    cores: dict[str, str] = {}
    for c in companies:
        for k in _company_core_keys(c):
            cores.setdefault(k.lower(), c)
    found: list[str] = []
    for core, comp in cores.items():
        if core and core in ql and comp not in found:
            found.append(comp)
    for abbr, core in _ENTITY_ABBREV.items():
        if abbr in ql:
            comp = cores.get(core)
            if comp and comp not in found:
                found.append(comp)
    return found if len(found) >= 2 else []


def _rebalance_comparison(hits: list[dict], entities: list[str], top_k: int) -> list[dict]:
    """对比题实体均衡保活：按核心实体分组各取 Top-M，再按分补齐到 top_k。"""
    groups: dict[str, list[dict]] = {e: [] for e in entities}
    other: list[dict] = []
    for h in hits:
        comp = _extract_answer_entity([h], h.get("content") or "", entities)
        (groups[comp] if comp in groups else other).append(h)
    m = max(1, top_k // len(entities))
    chosen: list[dict] = []
    seen: set[str] = set()
    for e in entities:
        if not groups[e]:
            logger.warning("对比实体相关信息不足: top_k=%d", top_k)
        for h in groups[e][:m]:
            if h["chunk_id"] in seen:
                continue
            seen.add(h["chunk_id"])
            chosen.append(h)
    for h in other:
        if len(chosen) >= top_k:
            break
        if h["chunk_id"] in seen:
            continue
        seen.add(h["chunk_id"])
        chosen.append(h)
    return chosen[:top_k]


# ---- 分级降级兜底（§11）----

def _real_hits(hits: list[dict]) -> int:
    """真实命中数：排除强注入块（relay 字段、exact 数字/引号硬插）与章节父块。"""
    return sum(
        1 for h in hits
        if not h.get("is_relay") and not h.get("is_exact_phrase")
        and h.get("chunk_type") != "section"
    )


def _needs_fallback(hits: list[dict], threshold: int, min_candidates: int) -> bool:
    """召回健康度评估：触发降级的条件（relay 保底/综述父块命中≥2 视为健康豁免）。

    指标题口径（2026-08-12 决策）：relay 存在即豁免不降级——字段索引精确指路，
    保底块确定正确（MRR=1.0），不因年份过滤放宽引入额外候选（precision 优先，
    非 recall）。非指标题无 relay（use_field_relay=False），不受该分支影响。
    """
    if not hits:
        return True
    if any(h.get("is_relay") for h in hits):
        return False  # relay 精确指路：真实命中低是保底正常态，不降级
    if sum(1 for h in hits if h.get("chunk_type") == "section") >= 2:
        return False  # 综述题父块命中 ≥2：父块是主体召回，健康
    if _real_hits(hits) < threshold:
        return True
    if len(hits) < min_candidates:
        return True
    return False


def _fallback_levels(fy: int | None, window: int) -> list[tuple[int, dict]]:
    """分级降级参数序列 [(scheme_level, hybrid_search 参数覆盖), ...]。

    L1 关短语硬插（留软路）→ L2 年份 ±window（子步 1）→ L2 无年份（子步 2）
    → L3 关实体预过滤 → L4 纯 dense。fy 为 None（趋势已放宽）时 L2 只保留无年份子步。
    """
    levels = [(1, {"use_phrase_hard_insert": False})]
    if fy is not None:
        levels.append((2, {"use_phrase_hard_insert": False, "fiscal_year": None,
                           "fiscal_years": [fy - window, fy, fy + window]}))
        levels.append((2, {"use_phrase_hard_insert": False, "fiscal_year": None,
                           "fiscal_years": None}))
    else:
        levels.append((2, {"use_phrase_hard_insert": False, "fiscal_year": None,
                           "fiscal_years": None}))
    levels.append((3, {"use_phrase_hard_insert": False, "fiscal_year": None,
                       "fiscal_years": None, "use_subject_filter": False}))
    levels.append((4, {"use_phrase_hard_insert": False, "fiscal_year": None,
                       "fiscal_years": None, "use_subject_filter": False, "dense_only": True}))
    return levels


def _fallback_reason(hits: list[dict], threshold: int) -> str:
    if not hits:
        return "empty"
    return f"real_hits={_real_hits(hits)}<{threshold}" if _real_hits(hits) < threshold else f"candidates={len(hits)}"


async def _search_plan(
    plan: RoutePlan,
    question: str,
    org_id: str,
    user_visibility: str,
    top_k: int,
    fiscal_year: int | None = None,
    rerank_candidates: int | None = None,
) -> list[dict]:
    """按路由计划执行混合检索；复杂/抽象/多跳问题加深召回；可指定年份过滤。

    多跳路径（sub_queries 非空）：按 sub_query_deps 拓扑序执行子查询，
    前序子查询的答案实体注入依赖查询（"该行"→具体公司名），最后统一合并；
    非多跳路径：改写查询优先，兜底原问题。
    """
    settings = get_settings()
    embedder = get_embedder()
    hyde_doc: str | None = None
    # 复杂问题检索加深（放大各路 top_k 与 rerank 候选）
    depth = (
        settings.retrieval_depth_scale
        if plan.complexity == "complex" or plan.intent in ("abstract", "multi_hop")
        else 1.0
    )

    async def _search_one(q: str) -> list[dict]:
        nonlocal hyde_doc
        direct_field = (
            settings.fields_enabled and not plan.sub_queries and not plan.evidence_aspects
            and plan.intent != "multi_hop"
            and await asyncio.to_thread(
                _direct_field_relay_eligible, question, org_id, user_visibility,
            )
        )
        # §15.5 P0 生产路由：分析/综合型问题关闭表格路与字段 relay——表格块在 RRF
        # 三路累加虚高抢占候选、relay 表格块顶位挤掉相关叙述块（§15.5 P0 实证）。
        # 判定三层：LLM intent(abstract/multi_hop) + 综述触发词 + 叙述型兜底
        # （无指标命中 → 答案在叙述块、table 负贡献，即使 intent=factual 也关 table）。
        # 综述触发词用公司实体感知版：公司名里的"发展"等不是意图词（浦发 17 题 relay
        # 全失根因，见 _summary_query_company_aware）
        summary_trigger = _summary_query_company_aware(q, org_id, user_visibility)
        analysis = (
            plan.intent in ("abstract", "multi_hop")
            or summary_trigger
            or _is_narrative_query(q)
        )
        # HyDE 场景分流（2026-08-11）：精确数值直查/引号短语/纯指标直查禁用——
        # 原句约束性强，假设文档反而稀释精确信号；分析/趋势/综述题保留
        use_hyde = plan.needs_hyde and not _hyde_disallowed(q, org_id, user_visibility)
        if use_hyde and hyde_doc is None:
            hyde_doc = await generate_hypothetical_document(question)
        # 指标别名扩召回：仅分析类（非指标）题拼接规范别名进 embedding——metric 题
        # relay 字段索引已精确覆盖，扩展别名会污染 extract_subject（残留别名混入主体
        # 串致 find_subject_company 失败 → relay 失效，探针实证 0.613→0）。
        # 扩展文本只用于稠密/稀疏向量；hybrid_search 内部（relay/subject/短语/rerank）
        # 一律用原问题 q，避免别名污染。
        q_expanded = _expand_query_aliases(q) if analysis else q
        # 查询侧取稠密+模型原生稀疏（text_type=query）。
        # 阻塞调用（嵌入 API / Qdrant / rerank）统一丢进线程池，避免占用事件循环
        # （否则单 worker 下并发问答会在检索段被串行化）。
        dense_list, sparse_list = await asyncio.to_thread(
            embedder.embed_texts_with_sparse, [q_expanded], text_type="query"
        )
        query_vector = dense_list[0]
        if use_hyde and hyde_doc:
            d_hyde, _ = await asyncio.to_thread(
                embedder.embed_texts_with_sparse, [hyde_doc], text_type="query"
            )
            # HyDE 稠密融合（原查询 0.7 / 假设文档 0.3）：保留原查询约束 + 补充泛化，
            # 优于单用 HyDE 向量（负向 case 主要来自 HyDE 编造内容带偏检索）
            w = settings.hyde_dense_weight
            query_vector = [
                (1.0 - w) * a + w * b for a, b in zip(query_vector, d_hyde[0])
            ]
        query_sparse = sparse_list[0] if sparse_list else None
        # IDF 动态开关：metric/指标型查询关闭文档侧 IDF（查询稀疏除以 IDF 还原语境）
        if query_sparse and plan.query_type == "metric":
            from app.retrieval.idf import neutralize_idf

            query_sparse = neutralize_idf(query_sparse)
        # §18 章节父块检索：综述/总结/分析型问题额外召回 section 父块并入 RRF——
        # 相关块分散多页时叶子单点命中率低，父块命中代表"章节整体相关"（父块路
        # 对综合类问题为正收益、对单点事实题无影响，见 ablation_report §18）
        # 已判定为复杂摘要的问题不能因改写后缺少触发词而失去 section 入口。
        # 保留旧触发路径；精确字段直查不因 abstract 误分类而扩大候选。
        use_parent = (analysis and summary_trigger) or (
            plan.intent == "abstract" and plan.complexity == "complex" and not direct_field
        )
        # 趋势/跨年对比题与对抗核实题放宽年份过滤。后者的问题年份可能属于
        # 待核实断言，而证据位于下一年度披露的报告中；将断言年份当文档年份会
        # 把正确公司报告完整过滤掉。精确事实题仍保留严格年份约束。
        fy = _effective_fiscal_year(q, fiscal_year)

        # 分级降级兜底（§11）：向量/路由参数只算一次，各级仅重跑 hybrid_search
        # 过滤参数（短语硬插→年份窗口→无年份→实体→纯 dense），达标即停并记日志。
        def _call(overrides: dict):
            params: dict = {
                "query_vector": query_vector,
                "org_id": org_id,
                "user_visibility": user_visibility,
                # 原问题 q：relay/subject/短语提取/rerank 全部基于原句，防别名污染
                "query_text": q,
                "top_k": top_k,
                "query_sparse": query_sparse,
                "recall_depth": depth,
                "fiscal_year": fy,
                "use_table_route": not analysis,
                "use_field_relay": not analysis or direct_field,
                "field_relay_exact_year": direct_field,
                "use_phrase_route": settings.phrase_route_enabled,
                "use_parent_route": use_parent,
                "use_adjacent_route": settings.adjacent_page_enabled and analysis,
                "use_document_section_coverage": (
                    settings.document_section_coverage_enabled and analysis
                ),
                "rerank_candidates": rerank_candidates,
            }
            params.update(overrides)
            return asyncio.to_thread(hybrid_search, **params)

        hits = await _call({})
        threshold = settings.fallback_real_hit_threshold
        min_cand = settings.fallback_min_candidates
        if not settings.fallback_enabled or not _needs_fallback(hits, threshold, min_cand):
            return hits
        log_fallback(q, "L0_trigger", _fallback_reason(hits, threshold),
                     _real_hits(hits), len(hits))
        prev = _real_hits(hits)
        prev_set = {h["chunk_id"] for h in hits}
        best = hits
        stall = 0
        for level, overrides in _fallback_levels(fy, settings.fallback_year_window):
            if level > settings.fallback_max_level:
                break
            candidate = await _call(overrides)
            cur = _real_hits(candidate)
            cur_set = {h["chunk_id"] for h in candidate}
            changed = cur_set != prev_set
            log_fallback(q, f"L{level}", "", cur, len(candidate))
            if not _needs_fallback(candidate, threshold, min_cand):
                logger.info("降级达标 L%d: q=%s real_hits=%d", level, safe_ref(q), cur)
                return candidate
            # 熔断：仅当"结果集实际变化但仍无改善"连续 2 级才停——空操作级
            # （无短语/无窗口年份数据）不计数，保证能推进到 L3 无年份/L4 纯 dense
            if settings.fallback_stall_stop and level >= 2:
                stall = stall + 1 if (cur <= prev and changed) else 0
                if stall >= 2:
                    log_fallback(q, "stop", "stall", cur, len(candidate))
                    logger.info("降级熔断 L%d: q=%s real_hits=%d", level, safe_ref(q), cur)
                    return candidate
            prev, prev_set = cur, cur_set
        return best

    # 多跳 DAG 路径：拓扑序执行子查询，前序答案实体注入依赖查询
    if plan.sub_queries:
        deps = plan.sub_query_deps or {}
        try:
            companies = list_field_companies(org_id, user_visibility)
        except Exception:  # noqa: BLE001  registry 未初始化时退化为无实体注入
            companies = []
        entities: dict[int, str] = {}
        per_group: dict[int, list[dict]] = {}
        for i in _topo_order(deps, len(plan.sub_queries)):
            q = plan.sub_queries[i]
            for d in deps.get(i, []):
                ent = entities.get(d)
                if ent:
                    injected = _inject_dependency_entity(q, ent)
                    if injected != q:
                        logger.info(
                            "子查询依赖注入 #%d: before=%s after=%s",
                            i,
                            safe_ref(q),
                            safe_ref(injected),
                        )
                        q = injected
            hits = await _search_one(q)
            per_group[i] = hits
            # 有依赖方需要产出中间实体：从自身命中块/查询提取（供下游注入）
            if deps.get(i):
                entities[i] = _extract_answer_entity(hits, q, companies)
        # 轻量证据图只在多跳题启用，并与文本候选做列表级合并；图无法替代原文召回。
        if plan.intent == "multi_hop" and settings.evidence_graph_enabled:
            try:
                from app.retrieval.evidence_graph import graph_evidence_hits

                graph_hits = await asyncio.to_thread(
                    graph_evidence_hits,
                    question,
                    org_id,
                    user_visibility,
                    limit=top_k,
                )
                if graph_hits:
                    per_group[-1] = graph_hits
            except Exception as exc:
                logger.warning("evidence graph recall skipped: type=%s", type(exc).__name__)
        # 分组保活：每子查询独立 Top-N 再拼接（对比/多跳题证据均衡）
        order = ([-1] if -1 in per_group else []) + _topo_order(
            deps, len(plan.sub_queries)
        )
        return _merge_grouped(per_group, order, top_k)

    # 精确字段式改写保护仅用于简单问题。复杂题保留原有语义扩展，
    # 不把多方面分析强制退回原句；租户/可见级别/年份过滤仍由检索入口执行。
    base_query = plan.rewritten_query or question
    if plan.complexity == "simple":
        base_query = _constraint_preserving_query(
            question, base_query, org_id, user_visibility,
        )

    # 非指标题方面覆盖：每方面独立召回，并在列表级先保活一个候选。
    # 某方面没有候选时仅补检索一次，禁止循环扩大成本。
    if settings.aspect_retrieval_enabled and plan.evidence_aspects:
        per_aspect: dict[int, list[dict]] = {}
        for index, aspect in enumerate(plan.evidence_aspects[:top_k]):
            # 完整问题保留主体、财年和否定/对比等强约束；方面用于聚焦其中一个子任务。
            # 仅检索方面的消融会丢失这些约束并显著降低候选覆盖率。
            aspect_query = f"{base_query}；重点证据方面：{aspect}"
            aspect_hits = await _search_one(aspect_query)
            if not aspect_hits:
                aspect_hits = await _search_one(f"{base_query}；查找与{aspect}直接相关的财报原文")
            per_aspect[index] = [
                {**hit, "evidence_aspect": aspect} for hit in aspect_hits
            ]
        covered = _merge_coverage_groups(
            per_aspect,
            list(range(len(per_aspect))),
            top_k,
        )
        if covered:
            return covered

    q = base_query
    hits = await _search_one(q)
    merged = _merge_blocks(hits, top_k)
    # 对比题实体均衡保活（八）：检测到 ≥2 核心实体时按实体分组各取 Top-M，
    # 防止单一实体高相关块占满 top-8、另一对比主体仅 1~2 条（信息单边失真）
    if settings.comparison_balance_enabled and len(merged) >= 2:
        try:
            companies = list_field_companies(org_id, user_visibility)
            entities = _detect_comparison_entities(q, companies)
            if len(entities) >= 2:
                merged = _rebalance_comparison(merged, entities, top_k)
                logger.info(
                    "对比实体均衡保活: entities=%d selected=%d",
                    len(entities),
                    len(merged),
                )
        except Exception:  # noqa: BLE001  registry 未初始化等场景跳过均衡
            pass
    return merged


def _best_score(blocks: list[dict]) -> float | None:
    """top 命中主分数：rerank 优先，其次 RRF 融合分。"""
    if not blocks:
        return None
    r = blocks[0]
    return r.get("rerank_score") if r.get("rerank_score") is not None else r.get("fused_score")


def _has_real_hits(blocks: list[dict]) -> bool:
    """年份过滤后是否含非字段 relay 的真实检索命中（年份回退判断依据）。"""
    return any(not b.get("is_relay") for b in blocks)


async def stream_answer(
    question: str,
    org_id: str,
    user_visibility: str,
    top_k: int | None = None,
    session_id: str = "",
    response_mode: str = "text",
    use_tools: bool = False,
    actor_id: str = "anonymous",
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
    # Legacy answer-cache entries are not versioned by context strategy. Do not
    # replay old chunk-only answers while the complete-page path is enabled.
    if (settings.semantic_cache_enabled and not settings.complete_page_context_enabled
            and not settings.context_whole_block_packing_enabled and not session_id):
        try:
            embedder = get_embedder()
            q_list, _ = await asyncio.to_thread(
                embedder.embed_texts_with_sparse, [question], text_type="query"
            )
            q_vec = q_list[0]
            cache_entry = answer_lookup(org_id, user_visibility, q_vec, q_text=question)
        except Exception as exc:
            logger.warning("答案缓存查询失败，走完整流程: type=%s", type(exc).__name__)
            cache_entry = None
    if cache_entry is not None:
        logger.info(
            "语义答案缓存命中: org=%s q=%s", safe_ref(org_id), safe_ref(question)
        )
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
            logger.warning("检测到 prompt 注入: q=%s", safe_ref(question))
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
    #    回退条件：年份过滤后无非 relay 的真实检索命中（字段索引 relay 恒注入，不能算"命中该年份"）
    t_retrieval = time.perf_counter()
    year = _extract_year(question)
    with make_span("answer.search", attributes={
        "year": year, "top_k": k, "query_chars": len(question),
    }):
        blocks = await _search_plan(plan, question, org_id, user_visibility, k, fiscal_year=year)
        year_fell_back = False
        if year and not _has_real_hits(blocks):
            blocks = await _search_plan(plan, question, org_id, user_visibility, k, fiscal_year=None)
            year_fell_back = True
    logger.info(
        "retrieval done: q=%s top_k=%d year=%s fell_back=%s %.0fms",
        safe_ref(question), len(blocks), year, year_fell_back,
        (time.perf_counter() - t_retrieval) * 1000,
    )
    # Page expansion has no new reranker score. Retain the original retrieval
    # confidence instead of silently dropping its warning after materialization.
    retrieval_best_score = _best_score(blocks)
    page_context = {"used": False, "reason": "NOT_APPLICABLE"}
    if (settings.complete_page_context_enabled and blocks
            and not any(block.get("is_relay") for block in blocks)):
        from app.retrieval.page_context_runtime import complete_page_evidence

        try:
            blocks, page_context = await complete_page_evidence(
                question, blocks, org_id=org_id, visibility=user_visibility,
                llm=get_llm(), max_chars=settings.complete_page_context_max_chars,
            )
        except PermissionError:
            yield _emit({"event": "error", "data": {
                "code": "evidence_access_denied", "message": "无法访问所需来源。",
                "request_id": get_trace_id(),
            }})
            yield _emit({"event": "done", "data": {"usage": {}}})
            return
    whole_pack = (settings.context_whole_block_packing_enabled
                  and plan.query_type != "metric" and not page_context["used"]
                  and not any(b.get("is_relay") for b in blocks))
    if settings.context_compression_enabled and blocks and not page_context["used"] and not whole_pack:
        from app.retrieval.compression import compress_context

        blocks = compress_context(
            question,
            blocks,
            block_max_chars=settings.context_block_max_chars,
            total_max_chars=settings.context_total_max_chars,
        )

    # 2.5 字段抽取检索：metric 类问题 → 查独立字段索引，命中则作为强证据前置并发出 field 事件
    field_evidence: list[dict] = []
    if settings.fields_enabled and plan.query_type == "metric":
        try:
            field_evidence = _lookup_field_evidence(question, org_id, user_visibility, year)
        except Exception as exc:
            logger.warning("字段抽取检索失败（忽略）: type=%s", type(exc).__name__)
            field_evidence = []
    if field_evidence:
        yield _emit({"event": "field", "data": {"fields": field_evidence}})
        # Whole-page selection already preserves primary source evidence. Keep
        # field events but do not append unbudgeted duplicate field text to it.
        if not page_context["used"]:
            blocks = [_field_block(ev) for ev in field_evidence] + blocks

    if whole_pack:
        from app.retrieval.context_packing import pack_context, pack_context_deduplicated

        packer = pack_context_deduplicated if settings.context_source_dedup_enabled else pack_context
        blocks = packer(blocks, settings.context_total_max_chars)

    yield _emit({
        "event": "meta",
        "data": {
            "org_id": org_id,
            "top_k": len(blocks),
            "intent": plan.intent,
            "complexity": plan.complexity,
            "needs_hyde": plan.needs_hyde,
            "evidence_aspects": plan.evidence_aspects,
            "context_compressed": settings.context_compression_enabled and not page_context["used"] and not whole_pack,
            "context_whole_packed": whole_pack,
            "context_source_dedup": whole_pack and settings.context_source_dedup_enabled,
            "page_context": page_context,
            "response_mode": response_mode,
            "tools_enabled": use_tools,
            "year": year,
            "year_fell_back": year_fell_back,
        },
    })

    for b in blocks:
        for ref in b.get('contained_citations', []):
            yield _emit({"event": "citation", "data": {
                **ref, "carrier_chunk_id": b.get('chunk_id'), "score": 0.0,
            }})
        yield _emit({
            "event": "citation",
            "data": {
                "doc_id": b.get("doc_id"),
                "chunk_id": b.get("chunk_id"),
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
    best = retrieval_best_score if page_context["used"] else _best_score(blocks)
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
    t_gen = time.perf_counter()
    with make_span("answer.generate", attributes={
        "model": getattr(llm, "_model", ""), "complexity": plan.complexity,
    }) as gen_span:
        try:
            if use_tools:
                from app.generation.tools import ToolContext, run_tool_agent

                async def _authorized_search(query: str, limit: int) -> list[dict[str, Any]]:
                    tool_plan = await route_query(query)
                    hits = await _search_plan(
                        tool_plan,
                        query,
                        org_id,
                        user_visibility,
                        limit,
                        fiscal_year=_extract_year(query),
                    )
                    return [
                        {
                            "doc_id": hit.get("doc_id"),
                            "doc_name": hit.get("doc_name"),
                            "section_path": hit.get("section_path"),
                            "page": hit.get("page"),
                            "content": str(hit.get("content") or "")[:1200],
                        }
                        for hit in hits
                    ]

                answer_text = (
                    await run_tool_agent(
                        llm,
                        messages,
                        ToolContext(
                            subject=actor_id,
                            org_id=org_id,
                            visibility=user_visibility,
                            search=_authorized_search,
                        ),
                    )
                    or ""
                )
                if answer_text:
                    yield _emit({"event": "token", "data": {"delta": answer_text}})
            if response_mode == "structured" and not answer_text:
                from app.generation.structured import (
                    generate_structured_answer,
                    render_structured_answer,
                    structured_payload,
                )

                structured = await generate_structured_answer(
                    llm,
                    question,
                    blocks,
                    history=history,
                )
                if structured is not None:
                    answer_text = render_structured_answer(structured)
                    yield _emit(
                        {
                            "event": "structured",
                            "data": structured_payload(structured, blocks),
                        }
                    )
                    # Keep legacy SSE consumers usable while exposing validated JSON.
                    yield _emit({"event": "token", "data": {"delta": answer_text}})
                else:
                    yield _emit(
                        {
                            "event": "warning",
                            "data": {"message": "结构化输出校验失败，已降级为普通文本。"},
                        }
                    )
            if not answer_text:
                async for delta in llm.stream_chat(messages):
                    answer_text += delta
                    yield _emit({"event": "token", "data": {"delta": delta}})
        except Exception as exc:
            had_error = True
            logger.error("llm stream failed: type=%s", type(exc).__name__)
            yield _emit({
                "event": "error",
                "data": {
                    "code": "generation_failed",
                    "message": "生成服务暂时不可用，请稍后重试。",
                    "request_id": get_trace_id(),
                },
            })
        if gen_span is not None:
            gen_span.set_attribute("chars", len(answer_text))
            gen_span.set_attribute("had_error", had_error)
    logger.info(
        "generation done: q=%s chars=%d had_error=%s %.0fms",
        safe_ref(question), len(answer_text), had_error,
        (time.perf_counter() - t_gen) * 1000,
    )

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
        except Exception as exc:
            logger.warning("问答记录落库失败（忽略）: type=%s", type(exc).__name__)

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
