"""外部 API 缓解缓存（Redis，可选）：嵌入结果缓存 + 语义答案缓存。

- 嵌入缓存：相同文本（归一化）的稠密+稀疏向量复用，key=`emb:{text_type}:{output_type}:{sha256}`
  （dense 与 dense&sparse 是不同 API 调用，稀疏结果不能通用，故 output_type 进 key）；
- 语义答案缓存：无 session_id 的单轮问答按"问题向量相似度 >= 阈值"命中后整份回放
  （meta/citations/tokens/grounding/warning），省掉 rerank + LLM 生成（最贵的两段）；
- Redis 不可用时所有函数静默返回未命中/不写入，不影响主流程。
"""

from __future__ import annotations

import hashlib
import json
import math
import threading

from app.core.config import get_settings
from app.core.logging import get_logger
from app.store.redisx import get_redis

logger = get_logger(__name__)

_embed_stats_lock = threading.Lock()
_embed_stats = {
    "hits": 0,
    "misses": 0,
    "bypassed": 0,
    "unavailable": 0,
}


def _record_embed_stat(name: str) -> None:
    with _embed_stats_lock:
        _embed_stats[name] += 1


def _norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _vec_cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _char_overlap(a: str, b: str) -> float:
    """字符 2-gram Dice 重合度（0~1）：用于向量命中后的文本重合度二次校验。"""
    def bigrams(s: str) -> set[str]:
        s = s.replace(" ", "")
        if len(s) < 2:
            return {s} if s else set()
        return {s[i : i + 2] for i in range(len(s) - 1)}

    ga, gb = bigrams(a), bigrams(b)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    return 2.0 * inter / (len(ga) + len(gb))


# ---------- 嵌入结果缓存 ----------

def embed_cache_get(
    text: str, text_type: str, output_type: str = "dense"
) -> tuple[list[float], dict | None] | None:
    if text_type == "document" and get_settings().embed_document_cache_ttl <= 0:
        _record_embed_stat("bypassed")
        return None
    r = get_redis()
    if r is None:
        _record_embed_stat("unavailable")
        return None
    try:
        key = f"emb:{text_type}:{output_type}:{hashlib.sha256(_norm(text).encode('utf-8')).hexdigest()}"
        raw = r.get(key)
        if not raw:
            _record_embed_stat("misses")
            return None
        data = json.loads(raw)
        _record_embed_stat("hits")
        return data["dense"], data.get("sparse")
    except Exception as e:
        _record_embed_stat("unavailable")
        logger.warning("嵌入缓存读取失败: %s", e)
        return None


def cache_metrics_snapshot() -> dict:
    """返回当前进程的嵌入缓存计数和 Redis 数据集快照。

    命中计数是进程级指标；Redis 键数与内存是实例级指标。扫描只在
    显式请求指标端点时执行，不进入问答和入库热路径。
    """
    with _embed_stats_lock:
        stats = dict(_embed_stats)
    lookups = stats["hits"] + stats["misses"]
    result = {
        "redis_available": False,
        "redis_total_keys": 0,
        "redis_used_memory_bytes": 0,
        "embed_query_keys": 0,
        "embed_document_keys": 0,
        "embed_cache_hits": stats["hits"],
        "embed_cache_misses": stats["misses"],
        "embed_cache_bypassed": stats["bypassed"],
        "embed_cache_unavailable": stats["unavailable"],
        "embed_cache_hit_rate": round(stats["hits"] / lookups, 4) if lookups else 0.0,
    }
    r = get_redis()
    if r is None:
        return result
    try:
        memory = r.info("memory")
        result.update(
            redis_available=True,
            redis_total_keys=int(r.dbsize()),
            redis_used_memory_bytes=int(memory.get("used_memory", 0)),
            embed_query_keys=sum(1 for _ in r.scan_iter(match="emb:query:*", count=1000)),
            embed_document_keys=sum(
                1 for _ in r.scan_iter(match="emb:document:*", count=1000)
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis 缓存指标读取失败: %s", exc)
    return result


def embed_cache_set(
    text: str, text_type: str, output_type: str, dense: list[float], sparse: dict | None
) -> None:
    settings = get_settings()
    ttl = (
        settings.embed_document_cache_ttl
        if text_type == "document"
        else settings.embed_cache_ttl
    )
    if ttl <= 0:
        return
    r = get_redis()
    if r is None:
        return
    try:
        key = f"emb:{text_type}:{output_type}:{hashlib.sha256(_norm(text).encode('utf-8')).hexdigest()}"
        r.set(key, json.dumps({"dense": dense, "sparse": sparse}, ensure_ascii=False), ex=ttl)
    except Exception as e:
        logger.warning("嵌入缓存写入失败: %s", e)


# ---------- 语义答案缓存 ----------

def _answer_bucket_key(org_id: str, user_visibility: str) -> str:
    return f"answer_cache:{org_id}:{user_visibility}"


def _load_bucket(org_id: str, user_visibility: str) -> list[dict]:
    r = get_redis()
    if r is None:
        return []
    try:
        raw = r.get(_answer_bucket_key(org_id, user_visibility))
        if not raw:
            return []
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("答案缓存读取失败: %s", e)
        return []


def answer_lookup(
    org_id: str,
    user_visibility: str,
    q_vec: list[float],
    q_text: str = "",
    min_overlap: float | None = None,
) -> dict | None:
    """按问题向量相似度查找缓存条目；命中返回可回放的整份数据。

    二次校验：向量命中后，若提供 q_text，还会比对问题文本字符重合度
    （2-gram Dice）——向量相似但文本差异大视为可疑（防御换嵌入模型/极端
    短问句导致的误命中）。
    """
    r = get_redis()
    if r is None:
        return None
    settings = get_settings()
    threshold = settings.semantic_cache_threshold
    min_overlap = settings.answer_cache_min_overlap if min_overlap is None else min_overlap
    q_norm = _norm(q_text) if q_text else ""
    best, best_sim = None, threshold
    for entry in _load_bucket(org_id, user_visibility):
        sim = _vec_cosine(q_vec, entry.get("q_vec") or [])
        if sim < best_sim:
            continue
        if q_norm and entry.get("q_norm") and _char_overlap(q_norm, entry["q_norm"]) < min_overlap:
            continue
        best, best_sim = entry, sim
    return best


def answer_store(org_id: str, user_visibility: str, entry: dict) -> None:
    """写入缓存条目：桶内追加并裁剪至上限，刷新 TTL。"""
    r = get_redis()
    if r is None:
        return
    try:
        settings = get_settings()
        bucket = _load_bucket(org_id, user_visibility)
        # 相同问题（归一化）直接替换，避免重复
        bucket = [e for e in bucket if e.get("q_norm") != entry.get("q_norm")]
        bucket.append(entry)
        max_entries = settings.answer_cache_max_entries
        if len(bucket) > max_entries:
            bucket = bucket[-max_entries:]
        r.set(
            _answer_bucket_key(org_id, user_visibility),
            json.dumps(bucket, ensure_ascii=False),
            ex=settings.answer_cache_ttl,
        )
    except Exception as e:
        logger.warning("答案缓存写入失败: %s", e)
