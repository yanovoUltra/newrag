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
from app.retrieval.reranker import get_reranker
from app.retrieval.rrf import rrf_fuse
from app.store import qdrant as qdrant_store

logger = get_logger(__name__)

# 章节级父块只作上下文扩展，不进召回候选（避免与叶子重复命中）
_EXCLUDE_PARENT = {"section"}

# 否定词模式："除了X之外/除X以外/排除X/不包括X"（X 为被否定实体）
_NEG_PATTERNS = (
    re.compile(r"除了(.+?)(?:之外|以外|外)[，,。;；]?"),
    re.compile(r"除(.+?)(?:之外|以外|外)[，,。;；]?"),
    re.compile(r"(?:排除|不包括|剔除)(.+?)[，,。;；]"),
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
    """metric 问题解析主体公司 → 返回目标文档 doc_id 列表；无法确定主体返回 None（不过滤）。"""
    from sqlalchemy import select

    from app.fields.subject import find_subject_company
    from app.models.entities import FinancialField
    from app.store.registry import get_session, list_field_companies

    companies = list_field_companies(org_id, user_visibility)
    if not companies:
        return None
    company = find_subject_company(query_text, companies)
    if not company:
        return None
    allowed = qdrant_store.visible_levels(user_visibility)
    with get_session() as s:
        doc_ids = set(
            s.scalars(
                select(FinancialField.doc_id).where(
                    FinancialField.company == company,
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


def hybrid_search(
    query_vector: list[float],
    org_id: str,
    user_visibility: str,
    query_text: str | None = None,
    top_k: int | None = None,
    query_sparse: dict | None = None,
    recall_depth: float = 1.0,
    fiscal_year: int | None = None,
    doc_ids: list[str] | None = None,
) -> list[dict]:
    """混合检索主入口。

    - 稠密路：dense top-50
    - 稀疏路：模型原生稀疏（text_type=query，由调用方传入 query_sparse）top-50
    - 表格路：chunk_type=table 稠密召回 top-20
    多路 RRF 融合 →（可选）rerank 精排 → 取 top_k。
    recall_depth>1 时按比例放大各路 top_k 与 rerank 候选数（复杂/抽象问题用）。
    fiscal_year 非 None 时按年份过滤（陈旧文档防护）。
    doc_ids 非空时按文档过滤（主体预过滤）；未显式给出时，metric 类问题自动解析主体。
    """
    settings = get_settings()
    k = top_k or settings.retrieval_top_k
    depth = max(recall_depth, 1.0)

    # 否定词感知：识别"除了X之外"→ 剔除被否定实体后改写查询（文本 + 向量）
    query_text, query_vector, query_sparse = _apply_negation(query_text, query_vector, query_sparse)

    # 主体公司预过滤：metric 问题是限制目标文档，消除跨公司噪声
    if not doc_ids and query_text:
        doc_ids = _resolve_subject_doc_ids(query_text, org_id, user_visibility)

    routes: list[list[dict]] = [
        qdrant_store.search_dense(
            query_vector=query_vector,
            org_id=org_id,
            user_visibility=user_visibility,
            top_k=int(settings.recall_dense_top_k * depth),
            exclude_chunk_types=list(_EXCLUDE_PARENT),
            fiscal_year=fiscal_year,
            doc_ids=doc_ids,
        ),
        qdrant_store.search_dense(
            query_vector=query_vector,
            org_id=org_id,
            user_visibility=user_visibility,
            top_k=int(settings.recall_table_top_k * depth),
            chunk_type="table",
            exclude_chunk_types=list(_EXCLUDE_PARENT),
            fiscal_year=fiscal_year,
            doc_ids=doc_ids,
        ),
    ]
    if query_text and query_sparse:
        routes.append(
            qdrant_store.search_sparse(
                query_sparse=query_sparse,
                org_id=org_id,
                user_visibility=user_visibility,
                top_k=int(settings.recall_sparse_top_k * depth),
                exclude_chunk_types=list(_EXCLUDE_PARENT),
                fiscal_year=fiscal_year,
                doc_ids=doc_ids,
            )
        )

    fused = rrf_fuse(routes, k=settings.rrf_k, top_n=int(settings.rerank_candidates * depth))
    # 字段索引回填召回：metric 问题注入来源 chunk 强候选（已含答案的表格）
    relay: list[dict] | None = None
    relay_ids: set[str] = set()
    if query_text and settings.fields_enabled:
        try:
            relay = _field_relay_candidates(query_text, org_id, user_visibility, fiscal_year)
        except Exception:  # noqa: BLE001
            relay = None
        if relay:
            relay_ids = {c["chunk_id"] for c in relay}
            fused = [r for r in fused if r["chunk_id"] not in relay_ids]
            fused = relay + fused

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
            fused.sort(key=lambda r: r["rerank_score"], reverse=True)

    allowed = qdrant_store.visible_levels(user_visibility)
    verified = [
        r for r in fused if r.get("org_id") == org_id and r.get("visibility") in allowed
    ]
    top = verified[:k]
    _attach_parent_context(top)
    return top


def _attach_parent_context(hits: list[dict]) -> None:
    """为命中叶子附加所属章节父块内容（命中→父块上下文扩展）。

    父块按 doc_id 分组批量拉取（fetch_payloads 按 doc_id 幂等点 id），
    每个命中 dict 增加 parent_content 字段供生成侧组装；无父块则为空串。
    """
    by_doc: dict[str, list[str]] = {}
    for r in hits:
        pid = r.get("parent_id")
        if pid:
            by_doc.setdefault(r["doc_id"], []).append(pid)
    payloads: dict[str, dict] = {}
    for doc_id, pids in by_doc.items():
        payloads.update(qdrant_store.fetch_payloads(doc_id, pids))
    for r in hits:
        pid = r.get("parent_id")
        p = payloads.get(pid) if pid else None
        r["parent_content"] = p.get("content", "") if p else ""
