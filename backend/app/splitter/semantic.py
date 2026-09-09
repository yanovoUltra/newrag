"""语义精切：对超长散文叶子，用嵌入模型计算相邻句子相似度，在主题断点处切分。

设计（结构优先、语义为辅）：
- 章节/表格边界已在 build_leaves 阶段处理，这里只精切"较长的连续散文叶子"（>= target tokens 且句子数足够）；
- 相邻句子相似度低于阈值（SEMANTIC_SPLIT_SIM_THRESHOLD）视为主题切换点；
- 切出的每一段仍受 CHUNK_MIN_TOKENS 下限约束，避免过度碎片化；
- mock 嵌入后端（无真实语义）或嵌入调用失败时原样返回，不影响主流程。
"""

from __future__ import annotations

import re

from app.core.config import get_settings
from app.core.logging import get_logger
from app.splitter.chunker import Chunk
from app.splitter.tokens import count_tokens

logger = get_logger(__name__)

# 中文句号/分号/感叹号/问号 + 英文标点断句（与 chunker._split_paragraph 一致）
_SENT_SPLIT = re.compile(r"(?<=[。；;!?！？])")


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if p]


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(x * x for x in b) ** 0.5 or 1.0
    return dot / (na * nb)


def _adjacent_sims(embedder, sentences: list[str]) -> list[float]:
    """批量嵌入句子，返回相邻句对余弦相似度（长度 = len-1）。"""
    vecs: list[list[float]] = []
    batch_size = getattr(embedder, "_batch_size", 16)
    for i in range(0, len(sentences), batch_size):
        # 注意：embed_texts 只返回稠密列表（单值）；embed_texts_with_sparse 才返回 (dense, sparse)
        dense = embedder.embed_texts(sentences[i : i + batch_size], text_type="document")
        vecs.extend(dense)
    return [_cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]


def _split_at_valleys(sentences: list[str], sims: list[float], settings) -> list[str]:
    """贪心聚合：低于阈值处断句，且每段不低于 min tokens。"""
    threshold = settings.semantic_split_sim_threshold
    min_tokens = settings.chunk_min_tokens
    parts: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for i, s in enumerate(sentences):
        buf.append(s)
        buf_tokens += count_tokens(s)
        is_boundary = i < len(sims) and sims[i] < threshold
        if is_boundary and buf_tokens >= min_tokens:
            parts.append("".join(buf))
            buf, buf_tokens = [], 0
    if buf:
        parts.append("".join(buf))
    return [p for p in parts if p.strip()]


def refine_leaves_semantically(leaves: list[Chunk], settings=None) -> list[Chunk]:
    """对可精切的叶子按语义边界切分；返回新叶子列表（顺序不变，seq 重排）。

    仅处理 chunk_type=text 且 token >= target、句子数 >= semantic_min_sentences 的叶子；
    其余叶子原样保留（含其 id，便于父块回链）。
    """
    settings = settings or get_settings()
    if not settings.semantic_chunk_refine:
        return leaves
    embedder = None
    try:
        from app.embed.embedder import get_embedder

        embedder = get_embedder()
    except Exception as exc:  # 嵌入不可用（如未配置密钥）时跳过精切
        logger.warning("semantic refine skipped（嵌入不可用）: type=%s", type(exc).__name__)
        return leaves
    if embedder.name == "mock":
        return leaves  # mock 无真实语义，避免随机切分

    out: list[Chunk] = []
    for leaf in leaves:
        if leaf.chunk_type != "text" or leaf.token_count < settings.chunk_target_tokens:
            out.append(leaf)
            continue
        sentences = _split_sentences(leaf.content)
        if len(sentences) < settings.semantic_min_sentences:
            out.append(leaf)
            continue
        try:
            sims = _adjacent_sims(embedder, sentences)
        except Exception as exc:
            logger.warning(
                "semantic refine embed 失败，保留原叶子: type=%s",
                type(exc).__name__,
            )
            out.append(leaf)
            continue
        parts = _split_at_valleys(sentences, sims, settings)
        if len(parts) <= 1:
            out.append(leaf)
            continue
        for seq, content in enumerate(parts, start=1):
            out.append(
                Chunk(
                    id=_new_chunk_id(),
                    doc_id=leaf.doc_id,
                    page=leaf.page,
                    section_path=leaf.section_path,
                    chunk_type="text",
                    content=content,
                    token_count=count_tokens(content),
                    seq=seq,
                    parent_id=None,  # 父块在 refine 之后统一构建
                )
            )
    return out


def _new_chunk_id() -> str:
    import uuid

    return uuid.uuid4().hex
