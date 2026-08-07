"""Qdrant 向量存储：collection 管理 + upsert + 权限 payload 过滤检索。"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from app.core.config import get_settings
from app.core.logging import get_logger
from app.embed.embedder import EMBED_DIM
from app.splitter.chunker import Chunk

logger = get_logger(__name__)

_client: QdrantClient | None = None

# 可见性等级：public < internal < restricted
VISIBILITY_LEVELS = {"public": 0, "internal": 1, "restricted": 2}


def visible_levels(user_visibility: str) -> list[str]:
    """用户可见的 visibility 集合（<= 用户等级）。"""
    level = VISIBILITY_LEVELS.get(user_visibility, 0)
    return [v for v, lv in VISIBILITY_LEVELS.items() if lv <= level]


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=get_settings().qdrant_url, timeout=30, check_compatibility=False)
    return _client


def ensure_collection() -> None:
    """创建 chunks collection：dense(1024, cosine) + 稀疏向量槽（阶段二启用）。"""
    client = get_client()
    settings = get_settings()
    if client.collection_exists(settings.qdrant_collection):
        try:
            vectors = client.get_collection(settings.qdrant_collection).config.params.vectors
            has_dense = isinstance(vectors, dict) and "dense" in vectors
        except Exception:
            has_dense = False
        if not has_dense:
            logger.warning("collection %s 向量配置不匹配，删除重建", settings.qdrant_collection)
            client.delete_collection(settings.qdrant_collection)
        else:
            return
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            "dense": models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )
    for field in ("org_id", "visibility", "chunk_type", "doc_id"):
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    logger.info("collection created: %s", settings.qdrant_collection)


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
            "content": chunk.content,
        }
        if fiscal_year is not None:
            payload["fiscal_year"] = fiscal_year
        if fiscal_quarter is not None:
            payload["fiscal_quarter"] = fiscal_quarter
        if chunk.parent_id:
            payload["parent_id"] = chunk.parent_id
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


def search_dense(
    query_vector: list[float],
    org_id: str,
    user_visibility: str,
    top_k: int,
    chunk_type: str | None = None,
) -> list[dict]:
    """单路稠密检索，强制注入权限 payload 过滤。"""
    client = get_client()
    must: list[models.Condition] = [
        models.FieldCondition(key="org_id", match=models.MatchValue(value=org_id)),
        models.FieldCondition(
            key="visibility",
            match=models.MatchAny(any=visible_levels(user_visibility)),
        ),
    ]
    if chunk_type:
        must.append(models.FieldCondition(key="chunk_type", match=models.MatchValue(value=chunk_type)))
    hits = client.query_points(
        collection_name=get_settings().qdrant_collection,
        query=query_vector,
        using="dense",
        query_filter=models.Filter(must=must),
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
                "section_path": p.get("section_path", ""),
                "chunk_type": p.get("chunk_type", "text"),
                "content": p.get("content", ""),
                "score": h.score,
                # 检索后二次校验（权限）
                "org_id": p.get("org_id"),
                "visibility": p.get("visibility"),
            }
        )
    return results


def search_sparse(
    query_sparse: dict,
    org_id: str,
    user_visibility: str,
    top_k: int,
    chunk_type: str | None = None,
) -> list[dict]:
    """稀疏路召回（Qdrant IDF modifier 全局加权 = BM25 风格），强制权限过滤。"""
    client = get_client()
    must: list[models.Condition] = [
        models.FieldCondition(key="org_id", match=models.MatchValue(value=org_id)),
        models.FieldCondition(
            key="visibility",
            match=models.MatchAny(any=visible_levels(user_visibility)),
        ),
    ]
    if chunk_type:
        must.append(models.FieldCondition(key="chunk_type", match=models.MatchValue(value=chunk_type)))
    hits = client.query_points(
        collection_name=get_settings().qdrant_collection,
        query=models.SparseVector(indices=query_sparse["indices"], values=query_sparse["values"]),
        using="sparse",
        query_filter=models.Filter(must=must),
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
                "section_path": p.get("section_path", ""),
                "chunk_type": p.get("chunk_type", "text"),
                "content": p.get("content", ""),
                "score": h.score,
                "org_id": p.get("org_id"),
                "visibility": p.get("visibility"),
            }
        )
    return results


def count_docs() -> int:
    client = get_client()
    try:
        res = client.count(collection_name=get_settings().qdrant_collection, exact=True)
        return res.count
    except Exception:
        return 0
