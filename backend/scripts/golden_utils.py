"""golden 指纹化工具：相关块以内容 sha256 引用，语料重建（chunk_id 变）后自动映射到新 id。

背景：chunk_id 为 uuid4 随机，语料重建后 golden 的 relevant_chunk_ids 全部失效。
方案 A（本模块）：golden 同时存 relevant_chunk_ids + relevant_fingerprints（内容指纹），
评测时用 FingerprintIndex 把指纹映射回当前语料的 chunk_id——只要文本未变，重建后无需重跑 golden。

指纹规范：sha256(norm(content))，norm 与嵌入缓存一致（空白折叠 + 小写），保证同一文本稳定标识。

用法：
    from golden_utils import FingerprintIndex, load_golden, resolve_rel
    idx = FingerprintIndex.build()          # 全量扫描 Qdrant 构建 指纹→chunk_id 映射
    questions = load_golden("golden/eval_set.json")
    rel = resolve_rel(q, idx)               # 指纹优先，缺失回退 relevant_chunk_ids
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402


def content_fingerprint(text: str) -> str:
    """内容指纹：与嵌入缓存 key 一致的归一化（空白折叠 + 小写）后 sha256。"""
    norm = " ".join((text or "").split()).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class FingerprintIndex:
    """Qdrant 全量 chunk 的内容指纹索引：fingerprint -> [chunk_id, ...]（同内容多块共享指纹）。"""

    def __init__(self) -> None:
        self._fp_to_ids: dict[str, list[str]] = {}
        self.n_chunks = 0

    @classmethod
    def build(cls, org_id: str | None = None, visibility: str | None = None) -> "FingerprintIndex":
        idx = cls()
        client = qdrant_store.get_client()
        settings = get_settings()
        offset = None
        while True:
            res = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=2000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in res[0]:
                pay = p.payload or {}
                cid = pay.get("chunk_id")
                content = pay.get("content")
                if not cid or content is None:
                    continue
                if org_id and pay.get("org_id") != org_id:
                    continue
                if visibility and pay.get("visibility") != visibility:
                    continue
                idx._fp_to_ids.setdefault(content_fingerprint(content), []).append(cid)
                idx.n_chunks += 1
            if res[1] is None:
                break
            offset = res[1]
        return idx

    def resolve(self, fingerprints: list[str] | None) -> set[str]:
        """指纹 → 当前语料 chunk_id 集合（缺失的指纹忽略）。

        2026-08-12 同指纹去重：同一内容的多副本块（签名页/重复表格等）本质等价，
        相关集只保留一个，避免"同指纹匹配 N 块"把 rel 撑大、稀释 NDCG/Rec。
        取 _fp_to_ids 中的首个 id（build 顺序稳定，进程内一致）。
        """
        ids: set[str] = set()
        for fp in fingerprints or []:
            lst = self._fp_to_ids.get(fp)
            if lst:
                ids.add(lst[0])
        return ids

    def missing(self, fingerprints: list[str] | None) -> int:
        return sum(1 for fp in (fingerprints or []) if fp not in self._fp_to_ids)

    def fingerprint_of(self, chunk_id: str) -> str | None:
        for fp, ids in self._fp_to_ids.items():
            if chunk_id in ids:
                return fp
        return None


def load_golden(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("questions") or []


def resolve_rel(q: dict, fp_index: FingerprintIndex | None) -> set[str]:
    """解析一题的相关集：指纹优先（语料重建免疫）；无指纹或映射全缺时回退 relevant_chunk_ids。"""
    fps = q.get("relevant_fingerprints") or []
    if fps and fp_index is not None:
        ids = fp_index.resolve(fps)
        if ids:
            return ids
    return set(q.get("relevant_chunk_ids") or [])
