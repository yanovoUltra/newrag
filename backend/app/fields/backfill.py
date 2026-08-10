"""字段 source_chunk_id 回填：为字段定位其来源叶子 chunk（同 doc 内容含 指标标签+数值 的块）。

供两条路径复用：
- ingest 主流程：切分（阶段 3）完成后自动回填当前文档字段 → 新上传文档不再依赖手动脚本；
- scripts/enrich_field_chunks.py：全量补跑历史文档（存量修复）。

定位规则（与 search 侧"字段索引回填召回"配套）：
- 优先：叶子块内容同时含 (指标标签, 数值)；
- 兜底：仅含数值的叶子块（大表别列/文本块）。
"""

from __future__ import annotations

from sqlalchemy import select

from app.core.logging import get_logger
from app.models.entities import FinancialField
from app.splitter.chunker import Chunk
from app.store.registry import get_session

logger = get_logger(__name__)


def _norm(text) -> str:
    return (
        (text or "")
        .replace(",", "")
        .replace("，", "")
        .replace(" ", "")
        .replace("\u00a0", "")
        .replace("\n", "")
    )


def _match_chunk_id(leaves: list[tuple[str, str, str]], n_raw: str, n_label: str) -> str | None:
    """在叶子块中定位字段来源 chunk：优先标签+数值同时命中，兜底仅数值。"""
    if n_raw and n_label:
        cid = next(
            (cid2 for cid2, _ct, nc in leaves if n_raw in nc and n_label in nc),
            None,
        )
        if cid:
            return cid
    if n_raw:
        return next((cid2 for cid2, _ct, nc in leaves if n_raw in nc), None)
    return None


def backfill_doc_fields(doc_id: str, chunks: list[Chunk]) -> int:
    """回填单个文档全部字段的 source_chunk_id（跳过已回填字段）。返回本次回填条数。"""
    leaves = [(c.id, c.chunk_type, _norm(c.content)) for c in chunks if c.chunk_type != "section"]
    if not leaves:
        return 0
    updated = 0
    with get_session() as s:
        fields = list(
            s.scalars(select(FinancialField).where(FinancialField.doc_id == doc_id)).all()
        )
        for f in fields:
            if f.source_chunk_id:
                continue
            cid = _match_chunk_id(leaves, _norm(f.raw), _norm(f.metric_label))
            if cid:
                f.source_chunk_id = cid
                updated += 1
        s.commit()
    return updated
