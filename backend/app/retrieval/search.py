"""阶段二混合检索：多路召回（稠密 + 稀疏 + 表格）→ RRF 融合 → 重排序 → 权限二次校验 → 父块上下文扩展。

召回增强（可配置）：
- 主体公司预过滤：metric 问题解析主体公司 → 限定目标文档，消除跨公司噪声；
- 字段索引回填召回：命中字段的来源 chunk 作为强候选注入，让数字汇总表显式召回；
- 否定词感知：识别"除了X之外/排除X" → 剔除被否定实体后改写查询（查询文本 + 向量），
  防止被否定实体（如 iPhone）的独占块顶入候选（对抗性评测 §7.2 B 类）。
"""

from __future__ import annotations

import re

from app.core.config import get_settings
from app.core.logging import get_logger
from app.fields.numeric import numeric_match_score, parse_numeric_constraints
from app.fields.phrases import extract_number_phrases, norm_number_phrase
from app.retrieval.reranker import get_reranker
from app.retrieval.rrf import rrf_fuse
from app.splitter.tokens import count_tokens
from app.store import qdrant as qdrant_store

logger = get_logger(__name__)

# 章节类型权重（七）：按 section_path/doc_name 关键词分类，低价值章节（目录/释义/
# 免责声明）高频词多、易虚高抢占 Top-K；核心财务/业务分析提权。仅 rerank 路径生效。
_SECTION_WEIGHT_RULES = (
    (1.2, ("主要财务指标", "财务报表", "合并资产负债表", "合并利润表", "合并现金流量表",
           "利润表", "资产负债表", "现金流量表", "财务数据", "补充财务资料", "审计报告",
           "每股收益", "净资产收益率")),
    (1.1, ("管理层讨论", "经营情况", "业务概要", "董事会报告", "经营分析",
           "主营业务", "业务发展", "经营业绩")),
    (1.0, ("风险管理", "风险提示", "风险因素", "风险控制", "内部控制", "风险指标")),
    (0.7, ("目录", "释义", "重要提示", "免责声明", "声明", "前瞻性陈述",
           "投资者关系", "公司信息")),
)


def _section_weight(r: dict) -> float:
    """候选块的章节类型权重（默认 1.0）。"""
    p = (((r.get("section_path") or "") + " " + (r.get("doc_name") or ""))).lower()
    for w, kws in _SECTION_WEIGHT_RULES:
        if any(k.lower() in p for k in kws):
            return w
    return 1.0

# 章节级父块只作上下文扩展，不进召回候选（避免与叶子重复命中）
# summary 摘要块同样只走专用摘要锚点路（启用时 doc 限定展开），不进主路——
# 否则关闭摘要时主路仍会召回摘要块（与未建摘要的基线状态不一致，P2-1 口径修正）
_EXCLUDE_PARENT = {"section", "summary"}

# 否定词模式："除了X之外/除X以外/排除X/不包括X"（X 为被否定实体）
_NEG_PATTERNS = (
    re.compile(r"除了(.+?)(?:之外|以外|外)[，,。;；]?"),
    re.compile(r"除(.+?)(?:之外|以外|外)[，,。;；]?"),
    re.compile(r"(?:排除|不包括|剔除)(.+?)[，,。;；]"),
)

# 精确短语模式：引号 / 中文引号 / 书名号 内容（对抗题"观点断言/制度名"的关键词锚点）
_EXACT_PHRASE_PATTERNS = (
    re.compile(r'"([^"]{3,60})"'),
    re.compile(r"“([^”]{3,60})”"),
    re.compile(r"《([^》]{3,60})》"),
)


def _extract_exact_phrases(text: str) -> list[str]:
    """提取精确短语：引号/书名号内容（去重，超长截前 24 字防稀释）。

    对抗题常见"引号断言 + 核实"结构，断言里的专有名词/制度名在原始查询
    稀疏向量中被整句权重稀释，导致相关块未召回——短语单独检索可锚定。
    """
    phrases: list[str] = []
    seen: set[str] = set()
    for pat in _EXACT_PHRASE_PATTERNS:
        for m in pat.finditer(text or ""):
            p = (m.group(1) or "").strip()
            if not p:
                continue
            if len(p) > 24:
                p = p[:24]
            if p not in seen:
                seen.add(p)
                phrases.append(p)
    return phrases


# 金融固定术语白名单："非X"是固定术语、不是否定（"非经常性损益""非息收入"等）。
# 防御"剔除(非经常性损益后的净利润)"被整体当作被否定实体而误删从句。
_NEG_TERM_WHITELIST = (
    "非经常性损益", "非经常性", "非系统性风险", "非保本", "非标", "非标债权",
    "非利息收入", "非息收入", "非存款类", "非流动资产", "非流动负债",
    "非控股股东", "非全资", "非金融", "非银", "非居民",
)


def _parse_negated_entity(text: str) -> str | None:
    """提取被否定实体（如"除了iPhone之外"→"iPhone"）；无否定模式返回 None。"""
    t = (text or "").strip()
    for pat in _NEG_PATTERNS:
        m = pat.search(t)
        if m:
            entity = (m.group(1) or "").strip()
            # "排除房地产板块后" → "房地产板块"（去掉从句尾缀）
            entity = re.sub(r"[后中内]$", "", entity)
            # 实体过长（>16 字符）视为误匹配长从句，放弃改写
            if entity and len(entity) <= 16:
                # 固定术语保护："剔除非经常性损益后的净利润"的实体是术语，非否定
                if any(w in entity for w in _NEG_TERM_WHITELIST):
                    return None
                return entity
    return None


def _remove_negation_clause(text: str) -> str:
    """剔除否定从句（"除了iPhone之外，"），保留正向主体。"""
    t = text or ""
    for pat in _NEG_PATTERNS:
        t = pat.sub("", t)
    t = re.sub(r"^[，,。;；\s]+", "", t)
    return t.strip() or (text or "")


def _apply_negation(
    query_text: str | None,
    query_vector: list[float],
    query_sparse: dict | None,
) -> tuple[str | None, list[float], dict | None]:
    """否定词感知查询改写：识别"除了X之外"→ 剔除否定从句后重新嵌入（文本 + 稠密 + 稀疏）。

    只对含否定模式的查询生效（罕见路径，额外一次嵌入调用可接受）；
    改写后查询不再携带被否定实体的语义/词法锚点，被否定实体的独占块自然下沉。
    """
    if not query_text:
        return query_text, query_vector, query_sparse
    neg = _parse_negated_entity(query_text)
    if not neg:
        return query_text, query_vector, query_sparse
    clean = _remove_negation_clause(query_text)
    if not clean or clean == query_text:
        return query_text, query_vector, query_sparse
    try:
        from app.embed.embedder import get_embedder

        dense, sparse = get_embedder().embed_texts_with_sparse([clean], text_type="query")
        logger.info("否定词感知改写: %.40s → %.40s", query_text, clean)
        return clean, dense[0], (sparse[0] if sparse else None)
    except Exception as e:  # noqa: BLE001
        logger.warning("否定词查询改写失败（忽略）: %s", e)
        return query_text, query_vector, query_sparse


def _resolve_subject_doc_ids(query_text: str, org_id: str, user_visibility: str) -> list[str] | None:
    """解析问题显式公司 → 返回目标文档列表；支持单主体与跨公司比较。"""
    from sqlalchemy import select

    from app.fields.subject import find_mentioned_companies, find_subject_company
    from app.models.entities import FinancialField
    from app.store.registry import get_session, list_field_companies

    companies = list_field_companies(org_id, user_visibility)
    if not companies:
        return None
    company = find_subject_company(query_text, companies)
    matched = [company] if company else find_mentioned_companies(query_text, companies)
    if not matched:
        return None
    allowed = qdrant_store.visible_levels(user_visibility)
    with get_session() as s:
        doc_ids = set(
            s.scalars(
                select(FinancialField.doc_id).where(
                    FinancialField.company.in_(matched),
                    FinancialField.org_id == org_id,
                    FinancialField.visibility.in_(allowed),
                )
            ).all()
        )
    return list(doc_ids) if doc_ids else None


def _field_relay_candidates(
    query_text: str, org_id: str, user_visibility: str, fiscal_year: int | None
) -> list[dict] | None:
    """字段索引回填召回：metric 问题命中字段后，把来源 chunk 作为强候选返回。

    返回与检索结果同构的候选 dict 列表；无命中或无法定位主体返回 None。
    """
    from app.fields.metrics import extract_metric_from_question
    from app.fields.subject import find_subject_company
    from app.store.registry import list_field_companies, query_fields

    key, q_year = extract_metric_from_question(query_text)
    if key is None:
        return None
    company = find_subject_company(query_text, list_field_companies(org_id, user_visibility))
    if company is None:
        return None
    target_year = q_year or fiscal_year
    rows = query_fields([key], org_id, user_visibility, year=target_year, company=company, limit=10)
    if not rows and target_year is not None:
        rows = query_fields([key], org_id, user_visibility, year=None, company=company, limit=10)
    # 按 doc 分组取来源 chunk payload
    by_doc: dict[str, list[str]] = {}
    meta: dict[str, dict] = {}
    for r in rows:
        if not r.source_chunk_id:
            continue
        by_doc.setdefault(r.doc_id, []).append(r.source_chunk_id)
        meta.setdefault(r.source_chunk_id, {
            "doc_id": r.doc_id, "doc_name": r.company, "page": r.page,
            "section_path": r.section_path, "score": 1.0, "fused_score": 1.0,
        })
    if not by_doc:
        return None
    payloads: dict[str, dict] = {}
    for doc_id, cids in by_doc.items():
        payloads.update(qdrant_store.fetch_payloads(doc_id, cids))
    out: list[dict] = []
    for cid in meta:
        pay = payloads.get(cid)
        if not pay:
            continue
        out.append({
            "chunk_id": cid,
            "doc_id": meta[cid]["doc_id"],
            "doc_name": meta[cid]["doc_name"],
            "page": meta[cid]["page"],
            "section_path": meta[cid]["section_path"],
            "chunk_type": pay.get("chunk_type", "table"),
            "content": pay.get("content", ""),
            "parent_id": pay.get("parent_id"),
            "score": meta[cid]["score"],
            "rerank_score": None,
            "fused_score": meta[cid]["fused_score"],
            "org_id": org_id,
            "visibility": user_visibility,
            "is_structural": False,
            "is_relay": True,  # 字段索引回填强候选标记：年份回退判断依赖它
        })
    return out or None


def _phrase_sparse_route(
    query_text: str,
    org_id: str,
    user_visibility: str,
    top_k: int,
    fiscal_year: int | None = None,
    fiscal_years: list[int] | None = None,
    doc_ids: list[str] | None = None,
) -> tuple[list[dict] | None, list[dict] | None]:
    """精确短语路由到稀疏路，返回 (soft_route, quote_hard_hits)。

    - soft_route：引号/书名号 + 数字单位全部短语的稀疏检索合并（入 RRF 软融合）；
    - quote_hard_hits：引号/书名号短语的稀疏 top-3（硬插候选，对齐数字短语精确路——
      "《工商银行2025年年报》"这类书名号/引号是强符号锚点，dense 会泛化掉边界）。

    返回与检索结果同构的候选列表；无短语或嵌入失败返回 (None, None)。
    """
    quote_phrases = _extract_exact_phrases(query_text)
    phrases = quote_phrases + extract_number_phrases(query_text)
    if not phrases:
        return None, None
    try:
        from app.embed.embedder import get_embedder

        _, sparses = get_embedder().embed_texts_with_sparse(phrases, text_type="query")
    except Exception as e:  # noqa: BLE001
        logger.warning("短语稀疏检索失败（忽略）: %s", e)
        return None, None
    soft: dict[str, dict] = {}
    quote_hard: dict[str, dict] = {}
    for phrase, sp in zip(phrases, sparses):
        hits = qdrant_store.search_sparse(
            query_sparse=sp,
            org_id=org_id,
            user_visibility=user_visibility,
            top_k=top_k,
            exclude_chunk_types=list(_EXCLUDE_PARENT),
            fiscal_year=fiscal_year,
            fiscal_years=fiscal_years,
            doc_ids=doc_ids,
        )
        for h in hits:
            cid = h["chunk_id"]
            if cid not in soft or h.get("score", 0.0) > soft[cid].get("score", 0.0):
                soft[cid] = h
        if phrase in quote_phrases:
            for h in hits[:3]:
                quote_hard.setdefault(h["chunk_id"], h)
    soft_route = list(soft.values()) or None
    quote_hard_hits = [h for h in quote_hard.values()][:3] or None
    return soft_route, quote_hard_hits


def _number_phrase_exact_hits(
    query_text: str,
    org_id: str,
    user_visibility: str,
    fiscal_year: int | None = None,
    doc_ids: list[str] | None = None,
) -> list[dict] | None:
    """数字+单位短语精确检索：离线倒排索引（number_phrase_index 表）硬插候选。

    流程：query 提取"数值+单位"短语 → 归一化（与构建侧同口径）→ SQLite 精确匹配
    → 取来源 chunk payload。embedding 对精确数字不敏感（1.23亿 vs 123百万 混淆），
    精确匹配补足；返回与检索结果同构的候选列表（is_exact_phrase=True 标记），
    由 hybrid_search 硬插候选池（relay 之下、RRF 之上）。索引未构建时返回 None。
    """
    from app.store.registry import query_number_phrase_index

    phrases = [norm_number_phrase(p) for p in extract_number_phrases(query_text)]
    if not phrases:
        return None
    try:
        rows = query_number_phrase_index(
            phrases,
            org_id,
            user_visibility,
            fiscal_year=fiscal_year,
            doc_ids=doc_ids,
            limit=get_settings().number_phrase_max_insert * 3,
        )
    except Exception as e:  # noqa: BLE001  索引未构建/registry 未初始化时降级跳过
        logger.warning("数字短语精确检索失败（忽略）: %s", e)
        return None
    if not rows:
        return None
    by_doc: dict[str, list[str]] = {}
    for cid, doc_id in rows:
        by_doc.setdefault(doc_id, []).append(cid)
    payloads: dict[str, dict] = {}
    for doc_id, cids in by_doc.items():
        payloads.update(qdrant_store.fetch_payloads(doc_id, cids))
    out: list[dict] = []
    for cid, doc_id in rows:
        pay = payloads.get(cid)
        if not pay:
            continue
        out.append({
            "chunk_id": cid,
            "doc_id": doc_id,
            "doc_name": pay.get("doc_name", ""),
            "page": pay.get("page", 0),
            "section_path": pay.get("section_path", ""),
            "chunk_type": pay.get("chunk_type", "text"),
            "content": pay.get("content", ""),
            "parent_id": pay.get("parent_id"),
            "score": 1.0,
            "rerank_score": None,
            "fused_score": 1.0,
            "org_id": org_id,
            "visibility": user_visibility,
            "is_structural": False,
            "is_exact_phrase": True,  # 数字短语精确命中标记（与 relay 区分）
        })
    return out[: get_settings().number_phrase_max_insert] or None


def hybrid_search(
    query_vector: list[float],
    org_id: str,
    user_visibility: str,
    query_text: str | None = None,
    top_k: int | None = None,
    query_sparse: dict | None = None,
    recall_depth: float = 1.0,
    fiscal_year: int | None = None,
    fiscal_years: list[int] | None = None,
    doc_ids: list[str] | None = None,
    use_table_route: bool = True,
    use_field_relay: bool = True,
    use_phrase_route: bool = False,
    use_phrase_hard_insert: bool = True,
    use_parent_route: bool = False,
    use_subject_filter: bool = True,
    dense_only: bool = False,
    rerank_candidates: int | None = None,
) -> list[dict]:
    """混合检索主入口。

    - 稠密路：dense top-50
    - 稀疏路：模型原生稀疏（text_type=query，由调用方传入 query_sparse）top-50
    - 表格路：chunk_type=table 稠密召回 top-20（use_table_route=False 时跳过，
      用于分析/综述型问题——表格路对非指标题是负贡献，见 ablation_report §15.3）
    - 父块路：chunk_type=section 稠密召回 top-N（use_parent_route=True 时启用，
      综述/总结类问题章节整体相关、叶子单点命中率低，父块命中并入 RRF）
    多路 RRF 融合 →（可选）rerank 精排 → 取 top_k。
    recall_depth>1 时按比例放大各路 top_k 与 rerank 候选数（复杂/抽象问题用）。
    fiscal_year/fiscal_years 非空时按年份过滤（陈旧文档防护；多值用于降级 L2 前后年窗口）。
    doc_ids 非空时按文档过滤（主体预过滤）；未显式给出时，metric 类问题自动解析主体。
    use_field_relay=False 时跳过字段索引回填（relay）——趋势/多跳/综述等非指标题
      含指标词（如"毛利率趋势"）会触发 relay 表格块顶位、挤掉相关叙述块（§15.5 P0 实证），
      调用方应按路由意图决定：指标型问题开 relay、分析/综合型关闭。

    降级兜底参数（§11 分级降级）：
    - use_phrase_hard_insert=False：L1，关引号/数字精确硬插，保留短语稀疏软路；
    - fiscal_years=[y-1,y,y+1] / fiscal_years=None：L2，年份窗口/无年份；
    - use_subject_filter=False：L3，关实体预过滤（全文档检索）；
    - dense_only=True：L4，只跑 dense 路（rerank 保留）。
    """
    settings = get_settings()
    k = top_k or settings.retrieval_top_k
    depth = max(recall_depth, 1.0)
    # L4 纯 dense 兜底：只保留稠密路（rerank 精排不降）
    if dense_only:
        use_table_route = False
        use_field_relay = False
        use_phrase_route = False
        use_parent_route = False

    # 否定词感知：识别"除了X之外"→ 剔除被否定实体后改写查询（文本 + 向量）
    query_text, query_vector, query_sparse = _apply_negation(query_text, query_vector, query_sparse)

    # 主体公司预过滤：metric 问题是限制目标文档，消除跨公司噪声（L3 关闭）
    if use_subject_filter and not doc_ids and query_text:
        doc_ids = _resolve_subject_doc_ids(query_text, org_id, user_visibility)

    routes: list[list[dict]] = [
        qdrant_store.search_dense(
            query_vector=query_vector,
            org_id=org_id,
            user_visibility=user_visibility,
            top_k=int(settings.recall_dense_top_k * depth),
            exclude_chunk_types=list(_EXCLUDE_PARENT),
            fiscal_year=fiscal_year,
            fiscal_years=fiscal_years,
            doc_ids=doc_ids,
        ),
    ]
    if use_table_route:
        routes.append(
            qdrant_store.search_dense(
                query_vector=query_vector,
                org_id=org_id,
                user_visibility=user_visibility,
                top_k=int(settings.recall_table_top_k * depth),
                chunk_type="table",
                exclude_chunk_types=list(_EXCLUDE_PARENT),
                fiscal_year=fiscal_year,
                fiscal_years=fiscal_years,
                doc_ids=doc_ids,
            )
        )
    # 章节父块路：综述/总结类问题章节整体相关——检索 section 父块并入 RRF（§18）。
    # 池重构后 top_k=12（父块12+叶子6=18 槽，降级鲁棒性更优）；父块命中后由
    # _attach_parent_context 附加其子级（父块自身 parent_id 为空，不递归附加）
    parent_hits: list[dict] = []
    if use_parent_route:
        parent_hits = qdrant_store.search_dense(
            query_vector=query_vector,
            org_id=org_id,
            user_visibility=user_visibility,
            top_k=int(settings.parent_route_top_k * depth),
            chunk_type="section",
            fiscal_year=fiscal_year,
            fiscal_years=fiscal_years,
            doc_ids=doc_ids,
        )
        routes.append(parent_hits)
    # 摘要锚点独立召回路（阶段四验证）：dense 检索 chunk_type=summary top-N。
    # 摘要块 RRF 单路分低（~1/(60+rank)）进不了 top-8 候选（实测 dense top20 有 4 个
    # summary 但 RRF 候选 top-8 无）——对标 §18 父块路教训，独立召回命中后展开为
    # 原文父块硬插候选 + boost。未开启时零开销。
    # P0 防跨文档污染（2026-08-12）：摘要块仅目标文档存在（如中芯），若主体解析失败
    # doc_ids 为空则整路跳过——否则其它公司问题会把中芯摘要块召回并顶入 top-8。
    summary_hits: list[dict] = []
    if settings.summary_enabled and use_parent_route and query_text and doc_ids:
        try:
            summary_hits = qdrant_store.search_dense(
                query_vector=query_vector,
                org_id=org_id,
                user_visibility=user_visibility,
                top_k=settings.summary_route_top_k,
                chunk_type="summary",
                fiscal_year=fiscal_year,
                fiscal_years=fiscal_years,
                doc_ids=doc_ids,
            )
        except Exception:  # noqa: BLE001 独立路失败不阻断主流程
            summary_hits = []
    if query_text and query_sparse:
        routes.append(
            qdrant_store.search_sparse(
                query_sparse=query_sparse,
                org_id=org_id,
                user_visibility=user_visibility,
                top_k=int(settings.recall_sparse_top_k * depth),
                exclude_chunk_types=list(_EXCLUDE_PARENT),
                fiscal_year=fiscal_year,
                fiscal_years=fiscal_years,
                doc_ids=doc_ids,
            )
        )
    # 精确短语路由到稀疏路：引号/书名号短语独立成路（对抗题"断言核实"锚定召回，§17）。
    # 返回 (软融合路, 引号短语硬插命中 top-3)——引号硬插在下方 exact 段与数字精确合并。
    quote_hard_hits: list[dict] | None = None
    if use_phrase_route and query_text:
        phrase_route, quote_hard_hits = _phrase_sparse_route(
            query_text,
            org_id,
            user_visibility,
            top_k=int(settings.phrase_route_top_k * depth),
            fiscal_year=fiscal_year,
            fiscal_years=fiscal_years,
            doc_ids=doc_ids,
        )
        if phrase_route:
            routes.append(phrase_route)

    # §17.4 实测：非指标题放大 rerank 候选（8→32）是负优化（NDCG@8 0.3068→0.2956，
    # 提升 22/下降 40）——放大候选引入噪声、把已进 top8 的相关块挤出/排序变差
    # （§13.2"候选越少质量越高"规律对非指标题同样成立）。生产恒用 rerank_candidates=8；
    # rerank_candidates 参数保留为评测对照扩展点（eval_ablation 显式传值）。
    base_cand = settings.rerank_candidates if rerank_candidates is None else rerank_candidates
    # 综述题池重构（§18）：parent_route_top_k=12 + 叶子候选 summary_leaf_candidates=6（18 槽）。
    # 探针实证：叶 8→6 仅损失 1 题相关叶入池，换来父块结构性进 top-8 的降级鲁棒性
    # （rerank 降级 RRF 直出时父块在第 7-8 槽，no-rerank top8 相关块 0.35→0.96）
    if use_parent_route and rerank_candidates is None:
        base_cand = settings.summary_leaf_candidates
    fused = rrf_fuse(
        routes,
        k=settings.rrf_k,
        top_n=int(base_cand * depth),
    )
    # 精确锚点硬插：数字+单位精确索引命中 + 引号/书名号短语稀疏 top-3（§18 升级）——
    # embedding 对精确数字/符号边界不敏感，精确匹配是确定性强信号，直接前置进候选池。
    # 插入位置在 RRF 之上、relay 之下（relay 是指标字段索引的更强确定性信号，保持 top 位）。
    # 索引未构建/短语不命中时自动降级跳过。
    exact_hits: list[dict] | None = None
    if use_phrase_route and use_phrase_hard_insert and query_text:
        try:
            exact_hits = _number_phrase_exact_hits(
                query_text, org_id, user_visibility,
                fiscal_year=fiscal_year, doc_ids=doc_ids,
            )
            if quote_hard_hits:
                if exact_hits is None:
                    exact_hits = []
                seen_hard = {h["chunk_id"] for h in exact_hits}
                exact_hits.extend(h for h in quote_hard_hits if h["chunk_id"] not in seen_hard)
        except Exception:  # noqa: BLE001
            exact_hits = None
        if exact_hits:
            exact_ids = {c["chunk_id"] for c in exact_hits}
            fused = [r for r in fused if r["chunk_id"] not in exact_ids]
            fused = exact_hits + fused
    # 字段索引回填召回：metric 问题注入来源 chunk 强候选（已含答案的表格）
    # use_field_relay=False（非指标题：趋势/多跳/综述/比较/对抗）跳过——指标词会触发
    # relay 表格块顶位、挤掉相关叙述块（§15.5 P0 实证：锚点后 top8 全是 relay table 块）
    relay: list[dict] | None = None
    relay_ids: set[str] = set()
    if query_text and settings.fields_enabled and use_field_relay:
        try:
            relay = _field_relay_candidates(query_text, org_id, user_visibility, fiscal_year)
        except Exception:  # noqa: BLE001
            relay = None
        if relay:
            relay_ids = {c["chunk_id"] for c in relay}
            fused = [r for r in fused if r["chunk_id"] not in relay_ids]
            fused = relay + fused

    # 父块强候选注入（§18.3）：父块路命中但 RRF 单路分低（~1/(60+rank)）进不了
    # rerank 候选（探针实证：相关父块在候选外 100%）——父块命中直接附加到候选，
    # 由 rerank 排序 + parent_route_boost 保底（父块=章节整体相关的强信号）
    if parent_hits:
        fused_ids = {f["chunk_id"] for f in fused}
        for ph in parent_hits:
            if ph["chunk_id"] not in fused_ids and ph["chunk_id"] not in relay_ids:
                fused.append(ph)
                fused_ids.add(ph["chunk_id"])

    # 摘要锚点展开（阶段四验证）：独立召回路（summary_hits）+ 候选内 summary 块统一
    # 展开——有 parent_id 的章节摘要块 → 拉原文父块替换进候选（摘要只做检索指针，
    # 原文父块进 rerank/生成，防摘要编造数字）；文档综述块（无 parent_id）保留。
    # 与召回路同口径 doc_ids 限定：主体解析失败时整段跳过（防跨文档污染，2026-08-12）。
    # 未开启/无 summary 命中时零开销。
    if settings.summary_enabled and use_parent_route and doc_ids:
        sum_blocks = summary_hits + [
            h for h in fused
            if h.get("chunk_type") == "summary" and h.get("doc_id") in set(doc_ids)
        ]
        if sum_blocks:
            pids_by_doc: dict[str, list[str]] = {}
            for h in sum_blocks:
                pid = h.get("parent_id")
                if pid:
                    pids_by_doc.setdefault(h.get("doc_id"), []).append(pid)
            expanded: list[dict] = []
            for doc_id, pids in pids_by_doc.items():
                try:
                    payloads = qdrant_store.fetch_payloads(doc_id, list(dict.fromkeys(pids)))
                except Exception:  # noqa: BLE001 摘要展开失败不阻断（候选保持原样）
                    payloads = {}
                for cid, pay in payloads.items():
                    expanded.append({
                        "chunk_id": cid, "doc_id": doc_id,
                        "doc_name": pay.get("doc_name"), "page": pay.get("page"),
                        "section_path": pay.get("section_path", ""),
                        "chunk_type": pay.get("chunk_type", "text"),
                        "content": pay.get("content", ""),
                        "parent_id": pay.get("parent_id"),
                        "score": 0.0, "fused_score": 0.0,
                        "org_id": pay.get("org_id"), "visibility": pay.get("visibility"),
                        "is_structural": bool(pay.get("is_structural")),
                        "is_summary_expanded": True,
                    })
            if expanded:
                fused_ids = {f["chunk_id"] for f in fused}
                for block in expanded:
                    if block["chunk_id"] not in fused_ids and block["chunk_id"] not in relay_ids:
                        fused.append(block)
                        fused_ids.add(block["chunk_id"])
                fused = [
                    h for h in fused
                    if not (h.get("chunk_type") == "summary" and h.get("parent_id"))
                ]

    # 分析型问题候选池 MMR 去重（§16 已删）：消融证明负优化（非指标 NDCG 0.3533→0.2852），
    # 相关集本身为同主题多块，去重反而删掉相关块。见 ablation_report §16.4。

    # 数值范围约束（六）：解析"指标+比较符+阈值"，候选块同族数值满足度 → 奖励分。
    # 仅约束题生效（无约束返回 []，零开销）；不做硬过滤只提权重，防误伤语义相关块。
    constraints = (
        parse_numeric_constraints(query_text)
        if settings.numeric_range_enabled and query_text
        else []
    )
    if constraints and fused:
        for r in fused:
            r["numeric_match"] = numeric_match_score(r.get("content") or "", constraints)

    if fused:
        reranker = get_reranker()
        docs = [r["content"] for r in fused]
        scores = reranker.rerank(query_text or "", docs)
        for r, s in zip(fused, scores):
            r["rerank_score"] = s
        # 关闭 rerank（none）时保留 RRF 顺序；启用时按 rerank 分降序
        if reranker.name != "none":
            # 字段回填保底：relay 块（字段索引精确指路的答案块）加固定分，
            # 防止被语义打分挤出 top_k（确定性索引 > 概率性打分，方案 1）
            if relay:
                for r in fused:
                    if r["chunk_id"] in relay_ids:
                        r["rerank_score"] += settings.field_relay_boost
            # 父块保底：父块命中（章节整体相关）加固定分，防止长文本被语义打分挤出
            if parent_hits:
                parent_ids = {h["chunk_id"] for h in parent_hits}
                for r in fused:
                    if r["chunk_id"] in parent_ids:
                        r["rerank_score"] += settings.parent_route_boost
            # 摘要锚点保底：摘要展开出的原文父块同样加分（与父块路同一语义信号）
            for r in fused:
                if r.get("is_summary_expanded"):
                    r["rerank_score"] += settings.parent_route_boost
            # 数值约束满足度奖励（β 低于核心检索分，防过度压分）
            if constraints:
                for r in fused:
                    if r.get("numeric_match"):
                        r["rerank_score"] += settings.numeric_range_boost * r["numeric_match"]
            # 章节类型权重（七）：核心财务 1.2 / 业务分析 1.1 / 风险 1.0 / 目录释义 0.7——
            # 低价值章节（目录/释义/免责）高频词多、易虚高，加权下压省出 Top-K 名额
            if settings.section_weighting_enabled:
                for r in fused:
                    w = _section_weight(r)
                    r["section_weight"] = w
                    r["rerank_score"] *= w
            fused.sort(key=lambda r: r["rerank_score"], reverse=True)
            # 指标直查的 relay 来源于 (公司, 指标, 年份) 结构化字段索引，是确定性答案块。
            # rerank 只负责排列其余语义候选，不得把概率性高分块压到 relay 之前。
            if relay_ids:
                fused.sort(key=lambda r: r["chunk_id"] not in relay_ids)
        else:
            # RRF 直出路径：数值约束满足度并入 fused_score 再排序（仅约束题生效，
            # 保持"无约束时 RRF 顺序不变"的契约）
            if constraints and any(r.get("numeric_match") for r in fused):
                for r in fused:
                    if r.get("numeric_match"):
                        r["fused_score"] = (r.get("fused_score") or 0.0) + settings.numeric_range_boost * r["numeric_match"]
                fused.sort(key=lambda r: r.get("fused_score") or 0.0, reverse=True)

    allowed = qdrant_store.visible_levels(user_visibility)
    verified = [
        r for r in fused if r.get("org_id") == org_id and r.get("visibility") in allowed
    ]
    top = verified[:k]
    _attach_parent_context(top)
    return top


def _norm_index_map(text: str) -> tuple[str, list[int]]:
    """去空白文本 + 归一化字符位置 → 原文位置映射（用于父块锚点定位）。"""
    norm: list[str] = []
    mapping: list[int] = []
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        norm.append(ch)
        mapping.append(i)
    return "".join(norm), mapping


def _mark_anchor(parent: str, child: str) -> tuple[str, int, int]:
    """在父块内容中定位命中子块正文，用 <hit>...</hit> 标记。

    子块可能只是父块的一部分（按段落切块），取子块前缀在父块中做最长前缀匹配
    （逐级缩短：64→40→24→16→12→8 字符），找到即标记；找不到返回原文与 (0,0)。
    """
    if not parent or not child:
        return parent or "", 0, 0
    parent_norm, pmap = _norm_index_map(parent)
    child_norm, _ = _norm_index_map(child)
    if not parent_norm or not child_norm:
        return parent, 0, 0
    for n in (min(64, len(child_norm)), 40, 24, 16, 12, 8):
        i = parent_norm.find(child_norm[:n])
        if i >= 0:
            start_o = pmap[i]
            end_o = pmap[i + n - 1] + 1
            marked = (
                parent[:start_o] + "<hit>" + parent[start_o:end_o] + "</hit>" + parent[end_o:]
            )
            return marked, start_o, end_o
    return parent, 0, 0


def _cap_around_anchor(parent: str, anchor_start: int, anchor_end: int, max_tokens: int) -> str:
    """超长父块按锚点截窗：标题前缀 + 锚点前后窗口；仍超限则从窗口尾部按比例裁剪。

    中文财报文本约 1 字/token，字符预算按 max_tokens 计（锚点前 1/4、后 1/2，
    另尽力带开头标题——章节归属信息）。锚点位于窗口前部，尾部裁剪不会丢命中区。
    """
    if count_tokens(parent) <= max_tokens:
        return parent
    head = parent[: min(anchor_start, 160)]
    before = min(anchor_start, max(0, max_tokens // 4))
    after = min(len(parent) - anchor_end, max(0, max_tokens // 2))
    window = head + "…" + parent[anchor_start - before : anchor_end + after]
    if count_tokens(window) <= max_tokens:
        return window
    tokens = count_tokens(window)
    keep_ratio = max(0.5, min(1.0, max_tokens / tokens))
    return window[: int(len(window) * keep_ratio)]


def _attach_parent_context(hits: list[dict]) -> None:
    """为命中叶子附加所属章节父块内容（命中→父块上下文扩展）。

    治理（2026-08-11）：锚点 <hit> 标记 + token 上限（parent_max_tokens，超长截
    锚点前后窗口）+ 父块级去重（同一父块多个命中只附一份全文，其余置空防重复占
    token）。父块按 doc_id 分组批量拉取；无父块则为空串。
    """
    by_doc: dict[str, list[str]] = {}
    for r in hits:
        pid = r.get("parent_id")
        if pid:
            by_doc.setdefault(r["doc_id"], []).append(pid)
    payloads: dict[str, dict] = {}
    for doc_id, pids in by_doc.items():
        payloads.update(qdrant_store.fetch_payloads(doc_id, pids))
    max_tokens = get_settings().parent_max_tokens
    seen_parents: set[tuple[str, str]] = set()
    for r in hits:
        pid = r.get("parent_id")
        p = payloads.get(pid) if pid else None
        if not p:
            r["parent_content"] = ""
            continue
        key = (r["doc_id"], pid)
        if key in seen_parents:
            r["parent_content"] = ""  # 同父块已附全文，本块不再重复
            continue
        seen_parents.add(key)
        parent_raw = p.get("content", "")
        marked, a_start, a_end = _mark_anchor(parent_raw, r.get("content") or "")
        r["parent_content"] = _cap_around_anchor(marked, a_start, a_end, max_tokens)
