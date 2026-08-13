"""索引维护（阶段四：增量索引）：对账 / 局部修复 / 文档级重索引。

背景：文档入库按 doc 独立执行（天然增量，无需重建全量 collection）。本脚本补齐
索引健康度管理——向量库与 registry 状态对账，发现不一致时按文档局部修复/重索引。
所有关键动作走 JSON 结构化日志（app.core.logging），方便排查数据不一致。

用法（backend/ 下，venv python）：
    python scripts/index_maintenance.py stats
        汇总：Qdrant 总点数 / registry 文档数 / 每 doc 点数 vs chunk_count
    python scripts/index_maintenance.py check
        对账：孤儿点（Qdrant 有 registry 无）、缺块（indexed 但点数 != chunk_count）、
        残留点（registry 非 indexed 但有向量）；不一致项输出 doc 元数据 + 块级抽样明细；
        有异常时退出码 1
    python scripts/index_maintenance.py restore --doc <doc_id> [--org default]
        局部修复：向量丢失/半写时，从 pipeline stage（layout/sections/chunks/vectors）
        重入 Qdrant——不重新解析/不调嵌入 API（run_ingest 幂等，stage 全命中直接入库）
    python scripts/index_maintenance.py reindex --doc <doc_id> --file <源文件路径> [--org default]
        全量重建：清 pipeline stage + 删向量点 + 重跑完整 ingest（切分/嵌入参数变更后用）
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from qdrant_client import models

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402
from app.store.registry import (  # noqa: E402
    get_document,
    init_db,
    list_documents,
)

logger = get_logger("index_maintenance")


def _load_docs() -> list:
    init_db()
    docs = list_documents(org_id=None, limit=10000)
    logger.info("registry 加载完成: %d 文档", len(docs))
    return docs


def _qdrant_doc_counts() -> tuple[dict[str, int], dict[str, int]]:
    """Qdrant 全量 scroll：按 doc_id 统计总点数与摘要点数。"""
    client = qdrant_store.get_client()
    settings = get_settings()
    counts: dict[str, int] = {}
    summary_counts: dict[str, int] = {}
    offset = None
    t0 = time.perf_counter()
    while True:
        res = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=2000,
            offset=offset,
            with_payload=["doc_id", "chunk_type"],
            with_vectors=False,
        )
        for p in res[0]:
            did = (p.payload or {}).get("doc_id")
            if did:
                counts[did] = counts.get(did, 0) + 1
                if (p.payload or {}).get("chunk_type") == "summary":
                    summary_counts[did] = summary_counts.get(did, 0) + 1
        if res[1] is None:
            break
        offset = res[1]
    logger.info("Qdrant 全量扫描完成: %d 点 / %d doc_id，耗时 %.1fs",
                sum(counts.values()), len(counts), time.perf_counter() - t0)
    return counts, summary_counts


def _qdrant_doc_chunk_ids(doc_id: str) -> list[str]:
    """滚动指定 doc 的全部 chunk_id（详细日志用，只取 id 字段）。"""
    client = qdrant_store.get_client()
    settings = get_settings()
    ids: list[str] = []
    offset = None
    while True:
        res = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=2000,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
        )
        for p in res[0]:
            cid = (p.payload or {}).get("chunk_id")
            if cid:
                ids.append(cid)
        if res[1] is None:
            break
        offset = res[1]
    return ids


def _sample(ids: list[str], n: int = 5) -> list[str]:
    return ids[:n]


def _doc_summary(d) -> str:
    return (
        f"doc_id={d.id} filename={d.filename!r} status={d.status} "
        f"org={d.org_id} visibility={d.visibility} fiscal_year={d.fiscal_year} "
        f"chunk_count={d.chunk_count} created={d.created_at}"
    )


def _count_matches(expected: int, total: int, summaries: int) -> bool:
    """兼容摘要功能上线前后两种 registry.chunk_count 口径。"""
    return expected == total or expected == total - summaries


def cmd_stats(_args) -> int:
    docs = _load_docs()
    counts, summary_counts = _qdrant_doc_counts()
    total_q = sum(counts.values())
    indexed = [d for d in docs if d.status == "indexed"]
    print(f"Qdrant 点数: {total_q}（{len(counts)} 个 doc_id）")
    print(f"registry 文档: {len(docs)}（indexed {len(indexed)} / 其他 {len(docs) - len(indexed)}）")
    print(f"{'doc_id':<36}{'状态':<10}{'registry块':<10}{'qdrant点数':<10}{'一致':<5}")
    bad = 0
    for d in sorted(docs, key=lambda x: x.created_at):
        q = counts.get(d.id, 0)
        ok = d.status == "indexed" and _count_matches(
            d.chunk_count or 0, q, summary_counts.get(d.id, 0)
        )
        if not ok and d.status == "indexed":
            bad += 1
        print(f"{d.id:<36}{d.status:<10}{d.chunk_count or 0:<10}{q:<10}{'OK' if ok else '!':<5}")
    orphan = set(counts) - {d.id for d in docs}
    if orphan:
        print(f"\n孤儿点 doc_id（registry 无此文档）: {len(orphan)}")
        for oid in sorted(orphan)[:20]:
            print(f"  {oid}（{counts[oid]} 点）")
    logger.info("stats 完成: qdrant_total=%d registry_docs=%d indexed=%d 不一致=%d 孤儿=%d",
                total_q, len(docs), len(indexed), bad, len(orphan))
    return 0


def cmd_check(_args) -> int:
    """对账：输出每个不一致项的详细日志（doc 元数据 + 块级抽样），便于排查。"""
    docs = _load_docs()
    counts, summary_counts = _qdrant_doc_counts()
    doc_ids = {d.id for d in docs}
    problems: list[str] = []

    for d in docs:
        q = counts.get(d.id, 0)
        expected = d.chunk_count or 0
        summaries = summary_counts.get(d.id, 0)
        if d.status == "indexed" and not _count_matches(expected, q, summaries):
            diff = q - expected
            chunk_ids = _qdrant_doc_chunk_ids(d.id) if q > 0 else []
            logger.error(
                "索引不一致[缺块/多块] %s 期望=%d 实际=%d 差值=%+d 抽样=%s",
                _doc_summary(d), expected, q, diff, _sample(chunk_ids),
            )
            problems.append(
                f"缺块: {d.id} {d.filename} registry={expected} qdrant={q}（restore 可修复）"
            )
        elif d.status != "indexed" and q > 0:
            chunk_ids = _qdrant_doc_chunk_ids(d.id)
            logger.warning(
                "索引不一致[残留点] %s 状态=%s 但 qdrant=%d 点 抽样=%s",
                _doc_summary(d), d.status, q, _sample(chunk_ids),
            )
            problems.append(
                f"残留点: {d.id} status={d.status} 但 qdrant={q} 点（reindex/手动清理）"
            )
        elif d.status == "indexed":
            logger.info(
                "索引一致[OK] %s（%d 点，其中摘要 %d）",
                _doc_summary(d), q, summaries,
            )

    orphan = set(counts) - doc_ids
    for oid in sorted(orphan):
        logger.error("索引不一致[孤儿点] doc_id=%s 点数=%d（registry 无此文档，建议清理）", oid, counts[oid])
        problems.append(f"孤儿点: {oid} {counts[oid]} 点（registry 无记录）")

    if problems:
        logger.error("对账完成: %d 个不一致项", len(problems))
        print(f"发现 {len(problems)} 个不一致：")
        for p in problems:
            print("  -", p)
        return 1
    logger.info("对账完成: 索引一致（无孤儿点/缺块/残留点）")
    print("索引对账一致：无孤儿点、无缺块、无残留点。")
    return 0


def cmd_restore(args) -> int:
    """从 pipeline stage 重入向量（run_ingest 幂等：stage 全命中则直接入库）。"""
    from app.pipelines.ingest import run_ingest

    init_db()
    d = get_document(args.doc)
    if not d:
        logger.error("restore 失败: doc_id=%s 不存在", args.doc)
        print(f"文档不存在: {args.doc}")
        return 1
    stage_dir = get_settings().resolved_pipeline_dir / d.id
    missing = [f for f in ("04_chunks.json", "05_vectors.json")
               if not (stage_dir / f).exists()]
    if missing:
        logger.error("restore 失败: doc_id=%s 缺 stage %s（目录=%s），需源文件走 reindex",
                     d.id, missing, stage_dir)
        print(f"缺少 stage（{stage_dir} 无 {missing}），无法 restore；需源文件走 reindex")
        return 1
    before = qdrant_store_count(d.id)
    logger.info("restore 开始: %s stage 完整，重入前 qdrant=%d 点", _doc_summary(d), before)
    t0 = time.perf_counter()
    res = run_ingest(d.id, Path("_unused_for_stage_restore.pdf"), d.org_id, d.visibility,
                     d.fiscal_year, d.fiscal_quarter, None)
    after = qdrant_store_count(d.id)
    logger.info("restore 完成: doc_id=%s 重入后=%d 点，耗时 %.1fs（阶段耗时见 ingest 日志）",
                d.id, after, time.perf_counter() - t0)
    print(f"完成：{res}（qdrant {before} → {after} 点）")
    return 0


def cmd_reindex(args) -> int:
    """清 stage + 删向量 + 全量重跑（切分/嵌入参数变更后）。需要源文件。"""
    from app.pipelines.ingest import run_ingest

    init_db()
    d = get_document(args.doc)
    if not d:
        logger.error("reindex 失败: doc_id=%s 不存在", args.doc)
        print(f"文档不存在: {args.doc}")
        return 1
    src = Path(args.file)
    if not src.exists():
        logger.error("reindex 失败: 源文件不存在 path=%s", src)
        print(f"源文件不存在: {src}")
        return 1
    stage_dir = get_settings().resolved_pipeline_dir / d.id
    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
        logger.info("reindex 已清 stage: %s", stage_dir)
    qdrant_store.delete_doc(d.id)
    logger.info("reindex 开始: %s 源文件=%s，已清 stage 与向量", _doc_summary(d), src)
    t0 = time.perf_counter()
    res = run_ingest(d.id, src, d.org_id, d.visibility, d.fiscal_year, d.fiscal_quarter, None)
    logger.info("reindex 完成: doc_id=%s 耗时 %.1fs（阶段耗时见 ingest 日志）",
                d.id, time.perf_counter() - t0)
    print(f"完成：{res}")
    return 0


def qdrant_store_count(doc_id: str) -> int:
    """该 doc 当前 Qdrant 点数（restore/reindex 前后对比用）。"""
    client = qdrant_store.get_client()
    res = client.count(
        collection_name=get_settings().qdrant_collection,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        ),
        exact=True,
    )
    return res.count


def main() -> int:
    parser = argparse.ArgumentParser(description="索引维护：对账/修复/重索引（阶段四增量索引）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats", help="汇总统计")
    sub.add_parser("check", help="对账（有异常退出码 1；不一致输出详细日志）")
    p_r = sub.add_parser("restore", help="从 stage 重入向量")
    p_r.add_argument("--doc", required=True)
    p_r.add_argument("--org", default=None)
    p_r2 = sub.add_parser("reindex", help="清 stage+删向量+全量重跑")
    p_r2.add_argument("--doc", required=True)
    p_r2.add_argument("--file", required=True, help="源文件绝对路径（data/uploads 下）")
    p_r2.add_argument("--org", default=None)
    args = parser.parse_args()
    fn = {"stats": cmd_stats, "check": cmd_check, "restore": cmd_restore, "reindex": cmd_reindex}[args.cmd]
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
