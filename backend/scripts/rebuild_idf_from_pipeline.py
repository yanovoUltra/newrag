"""从 pipeline 原始稀疏向量 + 新 idf.json 幂等重建 collection 稀疏权重。

为何不用 idf.py 的 _rewrite_points：
- 它读 qdrant 当前值再乘 IDF，若部分点已写过会二次叠加（非幂等，且已半写坏）。
- 本脚本以 pipeline 05_vectors.json（应用 IDF 前的原始稀疏）为唯一真源，
  对每个点重新乘新 IDF 后仅更新 sparse 命名向量（不触碰 dense 与 payload），可安全重跑。

用法（backend/ 下）：python scripts/rebuild_idf_from_pipeline.py
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from qdrant_client import models  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.retrieval.idf import idf_path  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402
from app.store.registry import init_db, list_documents  # noqa: E402


def _load_json_lenient(path: Path) -> dict:
    """先 json.load；失败则 raw_decode 取首个完整 JSON 对象（容忍尾部垃圾字节）。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        dec = json.JSONDecoder()
        obj, _ = dec.raw_decode(path.read_text(encoding="utf-8"))
        return obj


def main() -> int:
    settings = get_settings()
    client = qdrant_store.get_client()
    init_db()

    idf_raw = json.loads(idf_path().read_text(encoding="utf-8"))
    idf = {int(k): float(v) for k, v in idf_raw.items() if k != "_idf"}
    print(f"idf.json: {len(idf)} index types")

    # 只处理 registry 中已 indexed 的文档，跳过孤儿/半写 pipeline（版本替换后残留）
    indexed_ids = {d.id for d in list_documents(org_id=None, limit=100000) if d.status == "indexed"}
    print(f"indexed docs: {len(indexed_ids)}")

    pipeline_dir = settings.resolved_pipeline_dir
    if not pipeline_dir.exists():
        print(f"pipeline 目录不存在: {pipeline_dir}")
        return 1

    upd = []
    total = 0
    for doc_dir in sorted(pipeline_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        if doc_id not in indexed_ids:
            print(f"  skip {doc_id}: not indexed (孤儿/半写)")
            continue
        chunks_f = doc_dir / "04_chunks.json"
        vecs_f = doc_dir / "05_vectors.json"
        if not (chunks_f.exists() and vecs_f.exists()):
            continue
        chunks = _load_json_lenient(chunks_f)["chunks"]
        vecs = _load_json_lenient(vecs_f)
        sparse = vecs.get("sparse")
        if not sparse or len(sparse) != len(chunks):
            print(f"skip {doc_id}: sparse({len(sparse) if sparse else 0}) != chunks({len(chunks)})")
            continue
        for chunk, sv in zip(chunks, sparse):
            pid = uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:{chunk['id']}").hex
            inds, vals = sv["indices"], sv["values"]
            weighted = [
                (i, v * idf.get(i, 1.0)) for i, v in zip(inds, vals)
            ]
            weighted = [(i, v) for i, v in weighted if v > 0]
            if not weighted:
                continue
            weighted.sort(key=lambda t: t[1], reverse=True)
            upd.append(
                models.PointVectors(
                    id=pid,
                    vector={
                        "sparse": models.SparseVector(
                            indices=[i for i, _ in weighted],
                            values=[v for _, v in weighted],
                        )
                    },
                )
            )
            total += 1
        print(f"  {doc_id}: {len(sparse)} points")

    for i in range(0, len(upd), 64):
        client.update_vectors(
            collection_name=settings.qdrant_collection,
            points=upd[i : i + 64],
            timeout=120,
        )
    print(f"done: updated {total} sparse vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())