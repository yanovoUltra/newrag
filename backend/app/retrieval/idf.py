"""文档侧 IDF 重写（稀疏路治本）。

背景（见 ablation_report.md 结论 4 + 后续诊断）：模型原生稀疏缺失 IDF 修正，Qdrant 稀疏按
`DOT(query·doc)` 打分。财务报表的"结构表头/页眉块"（含 `2024`/`本集团`/`财务报表`/`人民币百万元`
等）在 query 里命中多个泛化 index，score 高但非答案所在，RRF 将其抬进 top_k，稀释 dense+sparse。

诊断还证明"只在查询端乘 IDF"无效（eval_idf 实测 0.090→0.083）：表头块 doc 侧未降权，累加 score 仍靠前。
**必须文档侧重写**：入库时对稀疏 index 乘 IDF，让高频结构 index 权重压低、低频判别 index 抬升。

IDF 从当前 collection 的稀疏向量统计（按 index 的文档频率 DF，每个点计 1 次），指数项与 BM25 一致：
    idf = ln( (N - df + 0.5) / (df + 0.5) + 1 )
生成的 index→idf 映射写 data/idf.json，入库管线在 upsert 前应用；`scripts/rebuild_idf_from_pipeline.py`
可离线以 pipeline 原始向量为真源幂等重建并重写已入库点（旧 `scripts/rebuild_idf.py` 非幂等已废弃）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.store import qdrant as qdrant_store

logger = get_logger(__name__)
_MODULE_IDF_KEY = "_idf"
_IDF_CACHE: dict[int, float] | None = None


def idf_path() -> Path:
    return get_settings().resolved_sparse_idf_path


def _load_raw() -> dict[int, float]:
    p = idf_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {int(k): float(v) for k, v in raw.items() if k != _MODULE_IDF_KEY}
    except Exception as e:
        logger.warning("idf.json 读取失败，视为空: %s", e)
        return {}


def _idf_map() -> dict[int, float]:
    global _IDF_CACHE
    if _IDF_CACHE is None:
        _IDF_CACHE = _load_raw()
    return _IDF_CACHE


def reset_cache() -> None:
    global _IDF_CACHE
    _IDF_CACHE = None


def build_idf(rewrite: bool = False) -> int:
    """统计当前 collection 的稀疏 index 文档频率 → 计算 IDF → 写 data/idf.json。

    rewrite=True 时同时重写已入库点的稀疏权重（应用 IDF）。
    返回 IDF index 种类数。
    """
    settings = get_settings()
    client = qdrant_store.get_client()
    df: dict[int, int] = {}
    n = 0
    points = []
    offset = None
    while True:
        res = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1000, offset=offset, with_payload=True, with_vectors=True,
        )
        for p in res[0]:
            n += 1
            sv = p.vector.get("sparse") if isinstance(p.vector, dict) else None
            if sv is not None:
                for idx in set(sv.indices):
                    df[idx] = df.get(idx, 0) + 1
            points.append(p)
        if res[1] is None:
            break
        offset = res[1]

    idf = {idx: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for idx, f in df.items()}
    payload = {str(k): v for k, v in idf.items()}
    payload[_MODULE_IDF_KEY] = {"docs": n, "index_types": len(idf)}
    idf_path().parent.mkdir(parents=True, exist_ok=True)
    idf_path().write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    reset_cache()
    logger.info("IDF built: %d docs, %d index types -> %s", n, len(idf), idf_path())

    if rewrite:
        _rewrite_points(client, settings.qdrant_collection, points, idf)
    return len(idf)


def _rewrite_points(client, collection: str, points: list, idf: dict[int, float]) -> None:
    """用 IDF 重写每个点的稀疏向量（保留 payload 与稠密向量）。"""
    from qdrant_client import models

    new_points = []
    for p in points:
        vec = p.vector
        if not isinstance(vec, dict) or "sparse" not in vec:
            continue
        sv = vec["sparse"]
        weighted = [
            (i, v * idf.get(i, 1.0)) for i, v in zip(sv.indices, sv.values)
        ]
        weighted = [(i, v) for i, v in weighted if v > 0]
        if not weighted:
            continue
        weighted.sort(key=lambda t: t[1], reverse=True)
        new_vec = dict(vec)
        new_vec["sparse"] = models.SparseVector(
            indices=[i for i, _ in weighted], values=[v for _, v in weighted]
        )
        new_points.append(
            models.PointStruct(id=p.id, vector=new_vec, payload=p.payload)
        )
    for i in range(0, len(new_points), 64):
        client.upsert(collection_name=collection, points=new_points[i : i + 64], timeout=120)
    logger.info("rewrote %d points sparse with IDF", len(new_points))


def apply_idf(sparse: dict | None) -> dict | None:
    """对文档侧稀疏 (indices, values) 应用 IDF；IDF 不存在或未启用时原样返回。"""
    if not sparse:
        return sparse
    settings = get_settings()
    if not settings.sparse_idf_enabled:
        return sparse
    idf = _idf_map()
    if not idf:
        return sparse
    weighted = [(i, v * idf.get(i, 1.0)) for i, v in zip(sparse["indices"], sparse["values"])]
    weighted = [(i, v) for i, v in weighted if v > 0]
    if not weighted:
        return sparse
    weighted.sort(key=lambda t: t[1], reverse=True)
    return {"indices": [i for i, _ in weighted], "values": [v for _, v in weighted]}


def neutralize_idf(sparse: dict | None) -> dict | None:
    """撤销文档侧 IDF 的影响（用于 metric/指标型查询的动态开关）。

    IDF 已乘在入库文档的稀疏向量上（apply_idf），检索时点乘默认带 IDF 加权。
    对 metrics/label 类查询 IDF 为负收益，故把查询稀疏按 IDF 逐 index 相除，
    还原"未乘 IDF"的点乘语境（RRF 只用 rank，排序可精确还原，见 eval_idf 无 IDF 对照法）。
    """
    if not sparse:
        return sparse
    idf = _idf_map()
    if not idf:
        return sparse
    weighted = [(i, v / idf.get(i, 1.0)) for i, v in zip(sparse["indices"], sparse["values"])]
    weighted = [(i, v) for i, v in weighted if v > 0]
    if not weighted:
        return sparse
    weighted.sort(key=lambda t: t[1], reverse=True)
    return {"indices": [i for i, _ in weighted], "values": [v for _, v in weighted]}