"""Qdrant 向量存储：collection 管理 + upsert + 权限 payload 过滤检索。"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from app.core.config import get_settings
from app.core.logging import get_logger
from app.embed.embedder import EMBED_DIM
from app.splitter.chunker import Chunk

logger = get_logger(__name__)

_client: QdrantClient | None = None
_CONTAINMENT_MAX_CHARS = 8192
_LOCALHOST_URL = re.compile(r"(?<=://)localhost(?=[:/]|$)", re.IGNORECASE)

# 可见性等级：public < internal < restricted
VISIBILITY_LEVELS = {"public": 0, "internal": 1, "restricted": 2}
_PAYLOAD_INDEXES = {
    "org_id": models.PayloadSchemaType.KEYWORD,
    "visibility": models.PayloadSchemaType.KEYWORD,
    "chunk_type": models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "fiscal_year": models.PayloadSchemaType.INTEGER,
    "page": models.PayloadSchemaType.INTEGER,
}


class CollectionSchemaMismatchError(RuntimeError):
    """现有集合与应用模式不兼容；必须显式迁移，禁止自动删库。"""


def _normalize_loopback_url(url: str) -> str:
    """避免 Windows 上 localhost 的 IPv6 回环连接超时；其它主机名保持不变。"""
    return _LOCALHOST_URL.sub("127.0.0.1", url, count=1)


def visible_levels(user_visibility: str) -> list[str]:
    """用户可见的 visibility 集合（<= 用户等级）。"""
    level = VISIBILITY_LEVELS.get(user_visibility, 0)
    return [v for v, lv in VISIBILITY_LEVELS.items() if lv <= level]


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        # timeout=120：大文档分批写入需更长写超时，避免负载高时 30s 默认超时中断
        url = _normalize_loopback_url(get_settings().qdrant_url)
        _client = QdrantClient(url=url, timeout=120, check_compatibility=False)
    return _client


def ensure_collection() -> None:
    """创建 chunks collection：dense(1024, cosine) + 稀疏向量槽（模型原生稀疏，无 IDF modifier）。

    稀疏向量为嵌入模型原生输出（词表 index + 语义权重），不叠加 IDF；
    检测到配置不匹配时拒绝继续，避免启动过程自动删除生产数据。
    """
    client = get_client()
    settings = get_settings()
    if client.collection_exists(settings.qdrant_collection):
        try:
            info = client.get_collection(settings.qdrant_collection)
        except Exception as exc:
            raise RuntimeError("无法读取 Qdrant collection 配置") from exc
        issues = _collection_schema_issues(info)
        if issues:
            joined = "; ".join(issues)
            logger.error("collection schema mismatch: %s", joined)
            raise CollectionSchemaMismatchError(
                f"Qdrant collection 模式不兼容: {joined}；请运行版本化迁移，现有集合未被修改"
            )
        _ensure_payload_indexes(client, settings.qdrant_collection, info)
        return
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            "dense": models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )
    for field, schema in _PAYLOAD_INDEXES.items():
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name=field,
            field_schema=schema,
        )
    logger.info("collection created: %s", settings.qdrant_collection)


def _collection_schema_issues(info) -> list[str]:
    params = info.config.params
    vectors = params.vectors
    dense = vectors.get("dense") if isinstance(vectors, dict) else None
    issues: list[str] = []
    if dense is None:
        issues.append("缺少 dense named vector")
    else:
        if getattr(dense, "size", None) != EMBED_DIM:
            issues.append(f"dense 维度不是 {EMBED_DIM}")
        if getattr(dense, "distance", None) != models.Distance.COSINE:
            issues.append("dense 距离不是 cosine")
    sparse_vectors = params.sparse_vectors
    sparse = sparse_vectors.get("sparse") if isinstance(sparse_vectors, dict) else None
    if sparse is None:
        issues.append("缺少 sparse named vector")
    elif getattr(sparse, "modifier", None) is not None:
        issues.append("sparse modifier 必须为空")
    return issues


def _ensure_payload_indexes(client: QdrantClient, collection: str, info) -> None:
    existing = set((getattr(info, "payload_schema", None) or {}).keys())
    for field, schema in _PAYLOAD_INDEXES.items():
        if field in existing:
            continue
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=schema,
        )
        logger.info("payload index created: %s.%s", collection, field)


def upsert_chunks(
    doc_id: str,
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    doc_name: str,
    org_id: str,
    visibility: str,
    fiscal_year: int | None,
    fiscal_quarter: int | None,
    sparse_vectors: list[dict] | None = None,
) -> int:
    """写入向量点；权限字段进 payload；稀疏向量（阶段二）可选。"""
    client = get_client()
    settings = get_settings()
    points = []
    for idx, (chunk, vec) in enumerate(zip(chunks, dense_vectors)):
        payload: dict[str, Any] = {
            "chunk_id": chunk.id,
            "doc_id": doc_id,
            "doc_name": doc_name,
            "page": chunk.page,
            "section_path": chunk.section_path,
            "chunk_type": chunk.chunk_type,
            "org_id": org_id,
            "visibility": visibility,
            "token_count": chunk.token_count,
            "seq": chunk.seq,
            "content": chunk.content,
            "index_schema_version": settings.index_schema_version,
            "embedding_model": settings.embedding_model,
            "parser_version": settings.parser_version,
            "chunker_version": settings.chunker_version,
            "sparse_scheme": settings.sparse_scheme,
        }
        if fiscal_year is not None:
            payload["fiscal_year"] = fiscal_year
        if fiscal_quarter is not None:
            payload["fiscal_quarter"] = fiscal_quarter
        if chunk.parent_id:
            payload["parent_id"] = chunk.parent_id
        if chunk.chunk_type == "table":
            # 表格结构化元数据（供过滤/展示）
            if chunk.table_headers is not None:
                payload["table_headers"] = chunk.table_headers
            if chunk.n_rows is not None:
                payload["n_rows"] = chunk.n_rows
            if chunk.n_cols is not None:
                payload["n_cols"] = chunk.n_cols
        # 结构块标记（财务表头/页眉/版式说明），检索时降权/排除
        if settings.structural_chunk_enabled:
            from app.parsers.structural import is_structural_chunk

            payload["is_structural"] = bool(is_structural_chunk(chunk.content))
        else:
            payload["is_structural"] = False
        vector: dict[str, Any] = {"dense": vec}
        if sparse_vectors is not None:
            sv = sparse_vectors[idx]
            vector["sparse"] = models.SparseVector(indices=sv["indices"], values=sv["values"])
        points.append(
            models.PointStruct(
                id=uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:{chunk.id}").hex,
                vector=vector,
                payload=payload,
            )
        )
    for i in range(0, len(points), 64):
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points[i : i + 64],
        )
    logger.info("upserted %d points for doc %s", len(points), doc_id)
    return len(points)


def delete_doc(doc_id: str) -> int:
    client = get_client()
    res = client.delete(
        collection_name=get_settings().qdrant_collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            )
        ),
    )
    logger.info("deleted doc %s points", doc_id)
    # UpdateStatus 枚举不可直接 int()，统一返回 1 表示执行成功
    return 1 if getattr(res, "status", None) else 0


def _apply_structural(results: list[dict]) -> list[dict]:
    """结构块置后降权：把 is_structural 块移到末尾（rank 降级，RRF 贡献变小）。
    原 exclude/off 多模式实测对召回无影响，已收敛为单一降权行为。"""
    settings = get_settings()
    if not settings.structural_chunk_enabled:
        return results
    stable = [r for r in results if not r.get("is_structural")] + [
        r for r in results if r.get("is_structural")
    ]
    return stable


def _dedup_diverse(results: list[dict]) -> list[dict]:
    """召回多样性：精确重复始终去重，普通块继续按包含关系去重。

    长表格可能达到数十万字符；对所有候选做无界的两两子串搜索会把一次
    Qdrant 毫秒级查询放大到几十秒。长块仅用完整内容哈希判重，普通块保留
    原有包含关系语义，使复杂度受控且不会因截断哈希误删不同证据。
    """
    settings = get_settings()
    if not settings.retrieval_diversity_enabled:
        return results
    kept: list[dict] = []
    kept_norms: list[str] = []
    exact_hashes: set[bytes] = set()

    def _norm(s: str) -> str:
        return s.replace(",", "").replace(" ", "").replace("\n", "")

    for r in results:
        content = _norm(r.get("content") or "")
        if len(content) < 40:
            kept.append(r)
            continue
        digest = hashlib.sha256(content.encode("utf-8")).digest()
        dup = digest in exact_hashes
        if not dup and len(content) <= _CONTAINMENT_MAX_CHARS:
            dup = any(
                content in kept_content or kept_content in content
                for kept_content in kept_norms
                if len(kept_content) <= _CONTAINMENT_MAX_CHARS
            )
        if not dup:
            kept.append(r)
            kept_norms.append(content)
            exact_hashes.add(digest)
    return kept


def _year_filter_condition(fiscal_year: int | None, fiscal_years: list[int] | None) -> models.Condition | None:
    """年份过滤条件：单值 MatchValue；多值（降级 L2 前后年窗口）MatchAny；均无则 None。"""
    if fiscal_years:
        return models.FieldCondition(key="fiscal_year", match=models.MatchAny(any=list(fiscal_years)))
    if fiscal_year is not None:
        return models.FieldCondition(key="fiscal_year", match=models.MatchValue(value=fiscal_year))
    return None


def search_dense(
    query_vector: list[float],
    org_id: str,
    user_visibility: str,
    top_k: int,
    chunk_type: str | None = None,
    exclude_chunk_types: list[str] | None = None,
    fiscal_year: int | None = None,
    fiscal_years: list[int] | None = None,
    doc_ids: list[str] | None = None,
) -> list[dict]:
    """单路稠密检索，强制注入权限 payload 过滤；可追加年份过滤（陈旧文档防护）与文档过滤（主体预过滤）。"""
    client = get_client()
    must: list[models.Condition] = [
        models.FieldCondition(key="org_id", match=models.MatchValue(value=org_id)),
        models.FieldCondition(
            key="visibility",
            match=models.MatchAny(any=visible_levels(user_visibility)),
        ),
    ]
    yf = _year_filter_condition(fiscal_year, fiscal_years)
    if yf is not None:
        must.append(yf)
    if doc_ids:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids)))
    if chunk_type:
        must.append(models.FieldCondition(key="chunk_type", match=models.MatchValue(value=chunk_type)))
    must_not = None
    if exclude_chunk_types:
        must_not = [
            models.FieldCondition(key="chunk_type", match=models.MatchAny(any=exclude_chunk_types))
        ]
    settings = get_settings()
    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        using="dense",
        query_filter=models.Filter(must=must, must_not=must_not),
        limit=top_k,
        search_params=models.SearchParams(hnsw_ef=settings.qdrant_hnsw_ef, exact=False),
        with_payload=True,
    ).points
    results = []
    for h in hits:
        p = h.payload or {}
        results.append(
            {
                "chunk_id": p.get("chunk_id"),
                "doc_id": p.get("doc_id"),
                "doc_name": p.get("doc_name"),
                "page": p.get("page"),
                "seq": p.get("seq"),
                "section_path": p.get("section_path", ""),
                "chunk_type": p.get("chunk_type", "text"),
                "content": p.get("content", ""),
                "parent_id": p.get("parent_id"),
                "score": h.score,
                # 检索后二次校验（权限）
                "org_id": p.get("org_id"),
                "visibility": p.get("visibility"),
                "is_structural": bool(p.get("is_structural")),
            }
        )
    return _dedup_diverse(_apply_structural(results))


def search_sparse(
    query_sparse: dict,
    org_id: str,
    user_visibility: str,
    top_k: int,
    chunk_type: str | None = None,
    exclude_chunk_types: list[str] | None = None,
    fiscal_year: int | None = None,
    fiscal_years: list[int] | None = None,
    doc_ids: list[str] | None = None,
) -> list[dict]:
    """稀疏路召回（模型原生稀疏：词表 index + 语义权重），强制权限过滤；可追加年份/文档过滤。"""
    client = get_client()
    must: list[models.Condition] = [
        models.FieldCondition(key="org_id", match=models.MatchValue(value=org_id)),
        models.FieldCondition(
            key="visibility",
            match=models.MatchAny(any=visible_levels(user_visibility)),
        ),
    ]
    yf = _year_filter_condition(fiscal_year, fiscal_years)
    if yf is not None:
        must.append(yf)
    if doc_ids:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids)))
    if chunk_type:
        must.append(models.FieldCondition(key="chunk_type", match=models.MatchValue(value=chunk_type)))
    must_not = None
    if exclude_chunk_types:
        must_not = [
            models.FieldCondition(key="chunk_type", match=models.MatchAny(any=exclude_chunk_types))
        ]
    hits = client.query_points(
        collection_name=get_settings().qdrant_collection,
        query=models.SparseVector(indices=query_sparse["indices"], values=query_sparse["values"]),
        using="sparse",
        query_filter=models.Filter(must=must, must_not=must_not),
        limit=top_k,
        with_payload=True,
    ).points
    results = []
    for h in hits:
        p = h.payload or {}
        results.append(
            {
                "chunk_id": p.get("chunk_id"),
                "doc_id": p.get("doc_id"),
                "doc_name": p.get("doc_name"),
                "page": p.get("page"),
                "seq": p.get("seq"),
                "section_path": p.get("section_path", ""),
                "chunk_type": p.get("chunk_type", "text"),
                "content": p.get("content", ""),
                "parent_id": p.get("parent_id"),
                "score": h.score,
                "org_id": p.get("org_id"),
                "visibility": p.get("visibility"),
                "is_structural": bool(p.get("is_structural")),
            }
        )
    return _apply_structural(results)


def fetch_payloads(doc_id: str, chunk_ids: list[str]) -> dict[str, dict]:
    """按 chunk_id 取 payload（点 id 幂等 = uuid5(doc_id:chunk_id)），用于父块上下文扩展。"""
    if not chunk_ids:
        return {}
    client = get_client()
    ids = [uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:{cid}").hex for cid in chunk_ids]
    pts = client.retrieve(
        collection_name=get_settings().qdrant_collection,
        ids=ids,
        with_payload=True,
    )
    return {p.payload.get("chunk_id"): p.payload for p in pts if p.payload}


def fetch_page_window(
    seeds: list[tuple[str, int, int]],
    org_id: str,
    user_visibility: str,
    *,
    radius: int = 1,
    limit: int = 120,
) -> list[dict]:
    """Fetch tenant-safe narrative chunks on the same/adjacent pages as seed hits."""
    if not seeds or limit <= 0:
        return []
    pages_by_doc: dict[str, set[int]] = {}
    seed_pages_by_doc: dict[str, list[tuple[int, int, int]]] = {}
    for rank, (doc_id, page, seq) in enumerate(seeds):
        if not doc_id or page < 1:
            continue
        pages = pages_by_doc.setdefault(doc_id, set())
        pages.update(range(max(1, page - radius), page + radius + 1))
        seed_pages_by_doc.setdefault(doc_id, []).append((rank, page, seq))
    client = get_client()
    output: list[dict] = []
    # A filtered scroll is ordered by point id, not by page proximity. Fetch a
    # bounded pool first, then rank same-page evidence ahead of adjacent pages;
    # otherwise a low id can consume the cap and hide the requested seed page.
    scan_cap = max(2_000, limit * 20)
    for doc_id, pages in pages_by_doc.items():
        offset = None
        scanned = 0
        while scanned < scan_cap:
            points, offset = client.scroll(
                collection_name=get_settings().qdrant_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="org_id", match=models.MatchValue(value=org_id)),
                        models.FieldCondition(
                            key="visibility",
                            match=models.MatchAny(any=visible_levels(user_visibility)),
                        ),
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                        models.FieldCondition(key="page", match=models.MatchAny(any=sorted(pages))),
                        models.FieldCondition(key="chunk_type", match=models.MatchValue(value="text")),
                    ]
                ),
                limit=min(512, scan_cap - scanned),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            scanned += len(points)
            for point in points:
                payload = point.payload or {}
                if payload.get("org_id") != org_id:
                    continue
                if payload.get("visibility") not in visible_levels(user_visibility):
                    continue
                output.append(payload)
            if offset is None or not points:
                break

    def proximity(payload: dict) -> tuple[int, int, int, str, int, str]:
        doc_id = str(payload.get("doc_id") or "")
        page = int(payload.get("page") or 0)
        seq = int(payload.get("seq") or 0)
        distance, seq_distance, seed_rank = min(
            (
                abs(page - seed_page),
                abs(seq - seed_seq) if page == seed_page else 10_000,
                rank,
            )
            for rank, seed_page, seed_seq in seed_pages_by_doc.get(
                doc_id, [(10_000, page, seq)]
            )
        )
        return (
            distance,
            seq_distance,
            seed_rank,
            doc_id,
            seq,
            str(payload.get("chunk_id") or ""),
        )
    output.sort(
        key=proximity
    )
    return output[:limit]


def fetch_document_sections(
    doc_ids: list[str],
    org_id: str,
    user_visibility: str,
    *,
    scan_limit: int = 5_000,
) -> list[dict]:
    """Fetch tenant-safe section parent chunks for a bounded set of documents."""
    if not doc_ids or scan_limit <= 0:
        return []
    client = get_client()
    points, offset = client.scroll(
        collection_name=get_settings().qdrant_collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="org_id", match=models.MatchValue(value=org_id)),
                models.FieldCondition(
                    key="visibility",
                    match=models.MatchAny(any=visible_levels(user_visibility)),
                ),
                models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids)),
                models.FieldCondition(key="chunk_type", match=models.MatchValue(value="section")),
            ]
        ),
        limit=scan_limit,
        with_payload=True,
        with_vectors=False,
    )
    output = [
        point.payload
        for point in points
        if point.payload
        and point.payload.get("org_id") == org_id
        and point.payload.get("visibility") in visible_levels(user_visibility)
    ]
    output.sort(
        key=lambda payload: (
            str(payload.get("doc_id") or ""),
            int(payload.get("page") or 0),
            str(payload.get("section_path") or ""),
            str(payload.get("chunk_id") or ""),
        )
    )
    if offset is not None:
        logger.warning(
            "document section coverage scan reached cap: docs=%d cap=%d",
            len(doc_ids),
            scan_limit,
        )
    return output


def fetch_section_children(
    parents: list[tuple[str, str]],
    org_id: str,
    user_visibility: str,
    *,
    limit: int = 60,
) -> list[dict]:
    """Expand ranked section parents to tenant-safe leaf chunks."""
    if not parents or limit <= 0:
        return []
    parent_rank = {pair: rank for rank, pair in enumerate(parents)}
    ordered_parents = list(dict.fromkeys((doc_id, parent_id) for doc_id, parent_id in parents if doc_id and parent_id))
    client = get_client()
    output: list[dict] = []
    # Query small parent batches in ranking order. A single MatchAny over 90
    # parents returns point-id order and can spend the scan cap on low-ranked
    # sections before children of the best semantic parents are seen.
    for batch_start in range(0, len(ordered_parents), 10):
        batch = ordered_parents[batch_start : batch_start + 10]
        by_doc: dict[str, list[str]] = {}
        for doc_id, parent_id in batch:
            by_doc.setdefault(doc_id, []).append(parent_id)
        for doc_id, parent_ids in by_doc.items():
            points, offset = client.scroll(
                collection_name=get_settings().qdrant_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="org_id", match=models.MatchValue(value=org_id)),
                        models.FieldCondition(
                            key="visibility",
                            match=models.MatchAny(any=visible_levels(user_visibility)),
                        ),
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                        models.FieldCondition(
                            key="parent_id",
                            match=models.MatchAny(any=parent_ids),
                        ),
                    ]
                ),
                limit=512,
                with_payload=True,
                with_vectors=False,
            )
            output.extend(
                point.payload
                for point in points
                if point.payload
                and point.payload.get("org_id") == org_id
                and point.payload.get("visibility") in visible_levels(user_visibility)
            )
            if offset is not None:
                logger.warning("section child batch reached scan cap: doc=%s", doc_id)
        if len(output) >= limit * 4:
            break
    output.sort(
        key=lambda payload: (
            parent_rank.get(
                (str(payload.get("doc_id") or ""), str(payload.get("parent_id") or "")),
                len(parent_rank),
            ),
            int(payload.get("seq") or 0),
            str(payload.get("chunk_id") or ""),
        )
    )
    return output[:limit]
