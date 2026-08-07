"""阶段二混合检索：多路召回（稠密 + 稀疏 + 表格）→ RRF 融合 → 重排序 → 权限二次校验。"""

from __future__ import annotations

from app.core.config import get_settings
from app.embed.sparse import sparse_embed_one
from app.retrieval.reranker import get_reranker
from app.retrieval.rrf import rrf_fuse
from app.store import qdrant as qdrant_store


def hybrid_search(
    query_vector: list[float],
    org_id: str,
    user_visibility: str,
    query_text: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """混合检索主入口。

    - 稠密路：dense top-50
    - 稀疏路：jieba 分词 → BM25 风格（Qdrant IDF modifier）top-50
    - 表格路：chunk_type=table 稠密召回 top-20
    多路 RRF 融合 →（可选）rerank 精排 → 取 top_k。
    """
    settings = get_settings()
    k = top_k or settings.retrieval_top_k

    routes: list[list[dict]] = [
        qdrant_store.search_dense(
            query_vector=query_vector,
            org_id=org_id,
            user_visibility=user_visibility,
            top_k=settings.recall_dense_top_k,
        ),
        qdrant_store.search_dense(
            query_vector=query_vector,
            org_id=org_id,
            user_visibility=user_visibility,
            top_k=settings.recall_table_top_k,
            chunk_type="table",
        ),
    ]
    if query_text:
        routes.append(
            qdrant_store.search_sparse(
                query_sparse=sparse_embed_one(query_text),
                org_id=org_id,
                user_visibility=user_visibility,
                top_k=settings.recall_sparse_top_k,
            )
        )

    fused = rrf_fuse(routes, k=settings.rrf_k, top_n=settings.rerank_candidates)
    if fused:
        reranker = get_reranker()
        docs = [r["content"] for r in fused]
        scores = reranker.rerank(query_text or "", docs)
        for r, s in zip(fused, scores):
            r["rerank_score"] = s
        # 关闭 rerank（none）时保留 RRF 顺序；启用时按 rerank 分降序
        if reranker.name != "none":
            fused.sort(key=lambda r: r["rerank_score"], reverse=True)

    allowed = qdrant_store.visible_levels(user_visibility)
    verified = [
        r for r in fused if r.get("org_id") == org_id and r.get("visibility") in allowed
    ]
    return verified[:k]


def retrieve(
    query_vector: list[float],
    org_id: str,
    user_visibility: str,
    top_k: int | None = None,
) -> list[dict]:
    """兼容入口：无查询文本时退化为单路稠密（阶段一行为）。"""
    return hybrid_search(
        query_vector=query_vector,
        org_id=org_id,
        user_visibility=user_visibility,
        query_text=None,
        top_k=top_k,
    )
