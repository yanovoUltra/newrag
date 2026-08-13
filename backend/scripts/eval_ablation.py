"""检索消融网格（B1）。

对比各检索配置在小型 golden 集上的 NDCG@8 / Rec@8 / Rec@1 / Prec@8 / MRR：
  1. dense-only         仅稠密路（排除 section/table），无 rerank
  2. sparse-only        仅模型原生稀疏路，无 rerank
  3. table-only         仅表格路，无 rerank
  4. dense+sparse       稠密+稀疏 RRF，无 rerank
  5. dense+sparse+table 三路 RRF，无 rerank
  6. 生产完整            三路 RRF + rerank（走 .env 配置的真实 rerank API）
注：BM25 词法基线与稠密/稀疏加权配比实验已删除——结论已固化（ablation_report.md）：
    BM25 对指标型（200 题 NDCG 0.014 vs sparse 0.17）与非指标题（142 题 0.103 vs sparse 0.167）
    均无贡献；加权配比无意义，本语料（数字指标型+中文）不重做。

用法（backend/ 下）：
    python scripts/eval_ablation.py [--top-k 8] [--org default] [--visibility public]
结果同时写入 eval_results 表（eval_name=ablation）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import golden_utils  # noqa: E402  (scripts/golden_utils.py：指纹化相关集解析)

from app.core.config import get_settings  # noqa: E402
from app.embed.embedder import get_embedder  # noqa: E402
from app.pipelines.answer import _extract_year, _is_trend_query, _search_plan  # noqa: E402
from app.retrieval.router import (  # noqa: E402
    RoutePlan,
    _is_summary_query,
    classify_query_type_keyword,
    generate_hypothetical_document,
)
from app.retrieval.rrf import rrf_fuse  # noqa: E402
from app.retrieval.search import hybrid_search  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402
from app.store.registry import init_db, save_eval_result  # noqa: E402

GOLDEN_FILE = SCRIPT_DIR / "golden" / "eval_set.json"
_EXCLUDE = {"section"}


async def _gen_hyde_docs(questions: list[str]) -> dict[str, str]:
    """批量生成 HyDE 假设文档（LLM 内部已有 FairSemaphore 并发限流）。"""
    outs = await asyncio.gather(*[generate_hypothetical_document(q) for q in questions])
    return {q: d for q, d in zip(questions, outs) if d}


def _expand_rel_with_parents(rel: set[str], corpus: dict[str, dict]) -> set[str]:
    """评测口径扩展：相关叶子的父块也计入相关集（§18 父块检索的公平计分依据）。"""
    out = set(rel)
    for cid in rel:
        pid = corpus.get(cid, {}).get("parent_id")
        if pid:
            out.add(pid)
    return out


def _norm(text: str) -> str:
    return (
        text.replace(",", "")
        .replace("，", "")
        .replace(" ", "")
        .replace("\u00a0", "")
        .replace("\n", "")
    )


def _load_corpus() -> dict[str, str]:
    client = qdrant_store.get_client()
    settings = get_settings()
    corpus: dict[str, str] = {}
    offset = None
    while True:
        res = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in res[0]:
            payload = p.payload or {}
            content = payload.get("content") or ""
            corpus[payload["chunk_id"]] = {
                "content": _norm(content),
                "raw": content,
                "doc_id": payload.get("doc_id"),
                "parent_id": payload.get("parent_id"),
            }
        if res[1] is None:
            break
        offset = res[1]
    return corpus


def _ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    ranked = ranked[:k]
    dcg, idcg = 0.0, 0.0
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    for i in range(1, min(len(ranked), len(relevant)) + 1):
        idcg += 1.0 / math.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def _precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & relevant) / k


def _mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


async def _prod_real(
    question: str,
    org_id: str,
    user_visibility: str,
    top_k: int,
    year: int | None,
) -> list[dict]:
    """真实生产路径（口径修复 2026-08-11）：调用 answer._search_plan——含指标别名扩召回、
    HyDE 场景分流+稠密融合、趋势放宽（不依赖 analysis）、分级降级兜底、对比实体均衡。

    无 LLM 路由：intent=factual / needs_hyde=_is_summary_query（生产路由的保守近似，
    _search_one 内部以 _is_summary_query 判定 analysis，贴近生产行为）。
    year 传原始提取年份，趋势放宽由 _search_one 内部处理。
    """
    qtype = classify_query_type_keyword(question)
    plan = RoutePlan(
        intent="factual",
        complexity="simple",
        query_type=qtype,
        needs_hyde=_is_summary_query(question),
    )
    return await _search_plan(plan, question, org_id, user_visibility, top_k, fiscal_year=year)


def main() -> int:
    parser = argparse.ArgumentParser(description="检索消融网格")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--org", default="default")
    parser.add_argument("--visibility", default="public")
    parser.add_argument(
        "--golden", default=None,
        help="golden 文件路径（默认 scripts/golden/eval_set.json 整合测试集）",
    )
    parser.add_argument(
        "--skip-metric", action="store_true",
        help="只评测非指标题（metric 已稳定 0.87，跳过可省 rerank API 调用）",
    )
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="RRF-only 口径：跳过全部'生产完整'列（不调用 rerank API，符合 rerank 评测禁令）",
    )
    parser.add_argument(
        "--only-prod", action="store_true",
        help="生产环境级全量（2026-08-11）：仅保留无 rerank 基线 + '真实路径'列（完全镜像生产"
             "_search_plan 管线，含本地/API rerank）。本地 rerank 在 CPU 上是串行瓶颈（每次"
             "~15-30s），消融全列每题 10 次 rerank 全量要十几个小时——本模式每题仅 1 次，"
             "全量约 1.5~2 小时，指标照常写入 eval_results。与 --no-rerank 互斥（本模式强制开 rerank）",
    )
    parser.add_argument(
        "--rrf-k", type=int, default=60,
        help="RRF 融合常数 k（k 值消融：小语料建议 20~40；默认 60 为原文启发式）",
    )
    parser.add_argument(
        "--hyde", action="store_true",
        help="对非指标题启用 HyDE 列（LLM 生成假设文档后嵌入检索，§18 HyDE 量化）",
    )
    parser.add_argument(
        "--split-metric", action="store_true",
        help="按指标题/非指标题分组输出（整体/指标/非指标三张汇总表）",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="并发检索题数（2026-08-11 并发化：嵌入/检索/rerank 均为 IO 等待，"
             "串行 299 题要十几分钟；并发 4 数分钟内完成。注：Windows 本机并发过高会"
             "耗尽临时端口（WinError 10048，qdrant 短连接 TIME_WAIT 堆积），默认取 4）",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.no_rerank and not args.only_prod:
        # RRF-only 口径（rerank 禁令）：禁用 rerank 后端 → hybrid_search 内部
        # reranker.name="none" 不排序、不调 API，但 RRF 融合/relay 注入/父块保底/
        # 年份回退等生产管线照常执行（生产近似列，非全置空）
        settings.rerank_backend = "none"
    if args.only_prod and args.no_rerank:
        print("--only-prod 与 --no-rerank 互斥：--only-prod 强制开 rerank，忽略 --no-rerank")
        args.no_rerank = False
    init_db()
    golden_file = Path(args.golden) if args.golden else GOLDEN_FILE
    golden = json.loads(golden_file.read_text(encoding="utf-8"))["questions"]
    print(f"golden: {golden_file}（{len(golden)} 题）")

    print("构建内容指纹索引（语料重建免疫：golden 指纹 → 当前 chunk_id）...")
    fp_index = golden_utils.FingerprintIndex.build(args.org, args.visibility)
    print(f"指纹索引: {fp_index.n_chunks} chunks, {len(fp_index._fp_to_ids)} 唯一指纹")

    print("加载语料（滚动全量 payload）...")
    corpus = _load_corpus()
    corpus_ids = list(corpus.keys())
    print(f"语料 chunk 数: {len(corpus_ids)}")

    items: list[tuple[str, set[str], bool]] = []
    rel_parents_by_q: dict[str, set[str]] = {}  # 综述题「相关父块召回率」分母
    for q in golden:
        is_metric = q.get("type", "metric") == "metric"
        if args.skip_metric and is_metric:
            continue
        rel = golden_utils.resolve_rel(q, fp_index)
        # P3 通用兜底：相关块必须来自题目目标文档（防指纹跨公司误映射——#5 比亚迪块教训）
        doc_id = q.get("doc_id")
        if doc_id and rel:
            rel = {cid for cid in rel if corpus.get(cid, {}).get("doc_id") == doc_id}
        # §18 评测口径：相关叶子的父块计入相关集（父块检索的公平计分依据；
        # 基线各列不含父块故不受影响，仅父块列有加分空间）
        # 综述题主指标「相关父块召回率」：相关块的 parent_id 集合（章节覆盖）
        rel_parents = {corpus.get(cid, {}).get("parent_id") for cid in rel}
        rel_parents.discard(None)
        rel = _expand_rel_with_parents(rel, corpus)
        if not rel:
            print(f"  [跳过] 相关集为空: {q['question']}")
            continue
        items.append((q["question"], rel, is_metric))
        rel_parents_by_q[q["question"]] = rel_parents
    scope_suffix = "（仅非指标）" if args.skip_metric else ""
    print(f"有效 golden 条目: {len(items)}/{len(golden)}{scope_suffix}\n")

    # §18 HyDE 量化：预生成非指标题的假设文档（LLM 批量，含并发限流）
    hyde_map: dict[str, str] = {}
    if args.hyde:
        hyde_qs = [q for q, _rel, m in items if not m]
        print(f"预生成 HyDE 假设文档（{len(hyde_qs)} 题非指标）...")
        hyde_map = asyncio.run(_gen_hyde_docs(hyde_qs))
        print(f"HyDE 生成成功 {len(hyde_map)}/{len(hyde_qs)}")

    settings.rrf_k = args.rrf_k

    embedder = get_embedder()
    sem = asyncio.Semaphore(args.concurrency)

    async def _eval_one(question: str, is_metric: bool) -> dict:
        """单题全列检索（并发化 2026-08-11）：嵌入/检索/rerank 均 IO 等待，
        全部丢进线程池，同一事件循环内并发跑多题（embed/LLM 内部已有 FairSemaphore 限流）。"""
        async with sem:
            raw_year = _extract_year(question)  # 原始年份：生产列由 _search_one 内部做趋势放宽
            year = raw_year
            # §17.4 与 answer.py 一致：分析类（非指标）趋势/跨年对比题放宽年份过滤
            # （单一 fiscal_year 截断跨年对比块，探针 3 题 >50→top5）——仅手动模拟列使用
            if not is_metric and year is not None and _is_trend_query(question):
                year = None
            dense, sparse = await asyncio.to_thread(
                embedder.embed_texts_with_sparse, [question], text_type="query"
            )
            d_vec, s_vec = dense[0], (sparse[0] if sparse else None)
            # §18 HyDE 列：假设文档嵌入（查询文本/rerank 仍用原问题，与 answer.py 一致）
            d_vec_h: list[float] | None = None
            s_vec_h: dict | None = None
            if args.hyde and hyde_map.get(question):
                dh, sh = await asyncio.to_thread(
                    embedder.embed_texts_with_sparse, [hyde_map[question]], text_type="query"
                )
                d_vec_h, s_vec_h = dh[0], (sh[0] if sh else None)

            dense_hits = await asyncio.to_thread(
                qdrant_store.search_dense,
                query_vector=d_vec, org_id=args.org, user_visibility=args.visibility,
                top_k=settings.recall_dense_top_k, exclude_chunk_types=list(_EXCLUDE),
                fiscal_year=year,
            )
            sparse_hits: list[dict] = []
            if s_vec:
                sparse_hits = await asyncio.to_thread(
                    qdrant_store.search_sparse,
                    query_sparse=s_vec, org_id=args.org, user_visibility=args.visibility,
                    top_k=settings.recall_sparse_top_k, exclude_chunk_types=list(_EXCLUDE),
                    fiscal_year=year,
                )
            table_hits = await asyncio.to_thread(
                qdrant_store.search_dense,
                query_vector=d_vec, org_id=args.org, user_visibility=args.visibility,
                top_k=settings.recall_table_top_k, chunk_type="table",
                exclude_chunk_types=list(_EXCLUDE), fiscal_year=year,
            )

            # 生产完整列：hybrid_search（含 rerank + 字段 relay），并模拟与 answer.py 一致的
            # 年份回退——年份过滤后无非 relay 真实命中 → 回退全量再检索（修复 Rec@8 口径）
            # §15.5 P0 生产路由：非指标题（abstract/multi_hop 等分析/综合类）关闭表格路与字段
            # relay——表格块在 RRF 三路累加虚高抢占候选、relay 表格块顶位（探针实证 0/2→2/2）；
            # depth>1：§15.5 P1 召回加深验证（复杂/多跳/综述 top_k 放大，RETRIEVAL_DEPTH_SCALE）
            async def _prod(use_table: bool, use_relay: bool, depth: float = 1.0,
                            cand: int | None = None, use_parent: bool = False,
                            use_phrase: bool = False, qvec: list[float] | None = None,
                            qsparse: dict | None = None) -> list[dict]:
                hy = await asyncio.to_thread(
                    hybrid_search,
                    query_vector=qvec or d_vec, org_id=args.org, user_visibility=args.visibility,
                    query_text=question, top_k=args.top_k, query_sparse=qsparse or s_vec,
                    fiscal_year=year, use_table_route=use_table, use_field_relay=use_relay,
                    use_parent_route=use_parent, use_phrase_route=use_phrase,
                    recall_depth=depth, rerank_candidates=cand,
                )
                if year and not any(not h.get("is_relay") for h in hy):
                    hy = await asyncio.to_thread(
                        hybrid_search,
                        query_vector=qvec or d_vec, org_id=args.org, user_visibility=args.visibility,
                        query_text=question, top_k=args.top_k, query_sparse=qsparse or s_vec,
                        fiscal_year=None, use_table_route=use_table, use_field_relay=use_relay,
                        use_parent_route=use_parent, use_phrase_route=use_phrase,
                        recall_depth=depth, rerank_candidates=cand,
                    )
                return hy

            # --only-prod（生产环境级全量，2026-08-11）：本地 rerank 在 CPU 上是串行瓶颈
            # （每次 ~15-30s），消融列每题 10 次 rerank 全量要十几个小时——只保留无 rerank
            # 基线 + 真实路径列（完全镜像生产 _search_plan 管线），其余混合列置空跳过。
            if args.only_prod:
                hyb = hyb_routed = hyb_route8 = hyb_notable = hyb_deep = []
                hyb_parent = hyb_phrase = hyb_hyde = hyb_all = []
            else:
                hyb_route8 = await _prod(is_metric, is_metric, cand=8)   # 显式候选 8（=生产默认，对照锚点）
                # 历史列：仅全量评测时计算（skip-metric 时省 rerank API 调用，占位复用锚点）
                hyb = hyb_route8 if args.skip_metric else await _prod(True, True)          # 现状：全开（指标题口径）
                hyb_routed = hyb_route8 if args.skip_metric else await _prod(is_metric, is_metric)
                hyb_notable = hyb_route8 if args.skip_metric else await _prod(False, True)  # 全关 table（P0-2 纯对照）
                hyb_deep = hyb_route8 if args.skip_metric else await _prod(is_metric, is_metric, depth=1.5)  # 召回加深（P1）
                # §18 三项召回端优化（仅非指标题评测启用，省 rerank API 调用）
                hyb_parent = await _prod(is_metric, is_metric, cand=8, use_parent=_is_summary_query(question))
                hyb_phrase = await _prod(is_metric, is_metric, cand=8, use_phrase=True)
                hyb_hyde = (
                    await _prod(is_metric, is_metric, cand=8, qvec=d_vec_h, qsparse=s_vec_h)
                    if args.hyde else hyb_route8
                )
                hyb_all = (
                    await _prod(is_metric, is_metric, cand=8,
                                use_parent=_is_summary_query(question), use_phrase=True,
                                qvec=d_vec_h, qsparse=s_vec_h)
                    if args.hyde else await _prod(is_metric, is_metric, cand=8,
                                                  use_parent=_is_summary_query(question), use_phrase=True)
                )
            # 真实生产路径（口径修复 2026-08-11）：直接调用 answer._search_plan——
            # 别名扩召回/HyDE 融合+分流/趋势放宽/降级兜底/实体均衡全量生效
            hyb_prod = await _prod_real(question, args.org, args.visibility, args.top_k, raw_year)
            print(f"  已检索: {question[:30]}")
            _report_progress()
            return {
                question: {
                    "dense": [h["chunk_id"] for h in dense_hits],
                    "sparse": [h["chunk_id"] for h in sparse_hits],
                    "table": [h["chunk_id"] for h in table_hits],
                    "dense_objs": dense_hits,
                    "sparse_objs": sparse_hits,
                    "table_objs": table_hits,
                    "hybrid": [h["chunk_id"] for h in hyb],
                    "hybrid_routed": [h["chunk_id"] for h in hyb_routed],
                    "hybrid_route8": [h["chunk_id"] for h in hyb_route8],
                    "hybrid_notable": [h["chunk_id"] for h in hyb_notable],
                    "hybrid_deep": [h["chunk_id"] for h in hyb_deep],
                    "hybrid_parent": [h["chunk_id"] for h in hyb_parent],
                    "hybrid_phrase": [h["chunk_id"] for h in hyb_phrase],
                    "hybrid_hyde": [h["chunk_id"] for h in hyb_hyde],
                    "hybrid_all": [h["chunk_id"] for h in hyb_all],
                    "hybrid_prod": [h["chunk_id"] for h in hyb_prod],
                }
            }

    t_retrieval = time.perf_counter()
    done_count = 0
    n_total = len(items)

    def _report_progress() -> None:
        """每完成 50 题打印一次进度 + 均耗 + 预计剩余（用户随时查日志查看）。"""
        nonlocal done_count
        done_count += 1
        if done_count % 50 == 0 or done_count == n_total:
            elapsed = time.perf_counter() - t_retrieval
            avg = elapsed / done_count
            remain = (n_total - done_count) * avg
            print(
                f"  [进度] {done_count}/{n_total} 题完成 | 已用 {elapsed/60:.1f}min | "
                f"均耗 {avg:.1f}s/题 | 预计剩余 {remain/60:.0f}min",
                flush=True,
            )

    async def _run_all() -> dict:
        # gather 必须在事件循环内调用（Py3.13 循环外调用会得到悬挂 future）
        outs = await asyncio.gather(*[_eval_one(q, m) for q, _rel, m in items])
        merged: dict[str, dict[str, list[str]]] = {}
        for o in outs:
            merged.update(o)
        return merged

    per_query = asyncio.run(_run_all())
    print(f"全部检索完成: {len(items)} 题，并发 {args.concurrency}，耗时 {time.perf_counter() - t_retrieval:.1f}s\n")

    def _eval(ranked: dict[str, list[str]], sub_items: list[tuple]) -> dict[str, dict]:
        """返回 {col: {ndcg, rec, rec1, prec, mrr, prel, per}}，基于 sub_items 子集。

        prel = 相关父块召回率@top_k（综述题主指标：相关父块入 top_k 比例，章节覆盖）。
        """
        out: dict[str, dict] = {}
        for col, ranks in ranked.items():
            ndcg = [_ndcg_at_k(ranks[q], rel, args.top_k) for q, rel, _ in sub_items]
            rec = [_recall_at_k(ranks[q], rel, args.top_k) for q, rel, _ in sub_items]
            rec1 = [_recall_at_k(ranks[q], rel, 1) for q, rel, _ in sub_items]
            prec = [_precision_at_k(ranks[q], rel, args.top_k) for q, rel, _ in sub_items]
            mrrs = [_mrr(ranks[q], rel) for q, rel, _ in sub_items]
            prel = [
                _recall_at_k(ranks[q], rel_parents_by_q.get(q, set()), args.top_k)
                for q, _rel, _ in sub_items
            ]
            out[col] = {
                "ndcg": sum(ndcg) / len(ndcg) if ndcg else 0.0,
                "rec": sum(rec) / len(rec) if rec else 0.0,
                "rec1": sum(rec1) / len(rec1) if rec1 else 0.0,
                "prec": sum(prec) / len(prec) if prec else 0.0,
                "mrr": sum(mrrs) / len(mrrs) if mrrs else 0.0,
                "prel": sum(prel) / len(prel) if prel else 0.0,
                "per": ndcg,
            }
        return out

    # 组装各列排序（skip-metric：省略冗余历史列，聚焦候选8锚点 + §18 三项优化）
    # --no-rerank 时 rerank 后端被禁（RRF+relay 直出），生产列照常评测（列名前缀区分）
    # --only-prod：仅基线（无 rerank）+ 真实路径（唯一生产级 rerank 列）
    prod_prefix = "生产(RRF+relay)" if args.no_rerank else "生产完整"
    ranked: dict[str, dict[str, list[str]]] = {
        "dense-only": {},
        "sparse-only": {},
        "table-only": {},
        "dense+sparse": {},
        "dense+sparse+table": {},
    }
    if not args.skip_metric and not args.only_prod:
        ranked.update({
            f"{prod_prefix}(+rerank)": {},
            f"{prod_prefix}(路由)": {},
            f"{prod_prefix}(无table)": {},
            f"{prod_prefix}(深度1.5)": {},
        })
    ranked[f"{prod_prefix}(真实路径)"] = {}   # 口径修复：直接调用 answer._search_plan
    if not args.only_prod:
        ranked.update({
            f"{prod_prefix}(路由,候选8)": {},
            f"{prod_prefix}(父块)": {},
            f"{prod_prefix}(数字短语)": {},
            f"{prod_prefix}(HyDE)": {},
            f"{prod_prefix}(全开)": {},
        })
    for q, rel, _ in items:
        d_ids = per_query[q]["dense"]
        s_ids = per_query[q]["sparse"]
        t_ids = per_query[q]["table"]
        ranked["dense-only"][q] = d_ids[: args.top_k]
        ranked["sparse-only"][q] = s_ids[: args.top_k]
        ranked["table-only"][q] = t_ids[: args.top_k]
        ds = rrf_fuse([per_query[q]["dense_objs"], per_query[q]["sparse_objs"]], k=args.rrf_k, top_n=args.top_k)
        dst = rrf_fuse(
            [per_query[q]["dense_objs"], per_query[q]["sparse_objs"], per_query[q]["table_objs"]],
            k=args.rrf_k, top_n=args.top_k,
        )
        ranked["dense+sparse"][q] = [h["chunk_id"] for h in ds]
        ranked["dense+sparse+table"][q] = [h["chunk_id"] for h in dst]
        if not args.skip_metric and not args.only_prod:
            ranked[f"{prod_prefix}(+rerank)"][q] = per_query[q]["hybrid"]
            ranked[f"{prod_prefix}(路由)"][q] = per_query[q]["hybrid_routed"]
            ranked[f"{prod_prefix}(无table)"][q] = per_query[q]["hybrid_notable"]
            ranked[f"{prod_prefix}(深度1.5)"][q] = per_query[q]["hybrid_deep"]
        ranked[f"{prod_prefix}(真实路径)"][q] = per_query[q]["hybrid_prod"]
        if not args.only_prod:
            ranked[f"{prod_prefix}(路由,候选8)"][q] = per_query[q]["hybrid_route8"]
            ranked[f"{prod_prefix}(父块)"][q] = per_query[q]["hybrid_parent"]
            ranked[f"{prod_prefix}(数字短语)"][q] = per_query[q]["hybrid_phrase"]
            ranked[f"{prod_prefix}(HyDE)"][q] = per_query[q]["hybrid_hyde"]
            ranked[f"{prod_prefix}(全开)"][q] = per_query[q]["hybrid_all"]

    results = _eval(ranked, items)
    # 指标题/非指标题分组（--split-metric）
    results_metric: dict[str, dict] | None = None
    results_non: dict[str, dict] | None = None
    if args.split_metric:
        results_metric = _eval(ranked, [it for it in items if it[2]])
        results_non = _eval(ranked, [it for it in items if not it[2]])

    ordered = [
        "dense-only", "sparse-only", "table-only", "dense+sparse", "dense+sparse+table",
    ]
    if not args.skip_metric and not args.only_prod:
        ordered += [f"{prod_prefix}(+rerank)", f"{prod_prefix}(路由)", f"{prod_prefix}(无table)", f"{prod_prefix}(深度1.5)"]
    ordered += [f"{prod_prefix}(真实路径)"]
    if not args.only_prod:
        ordered += [f"{prod_prefix}(路由,候选8)", f"{prod_prefix}(父块)", f"{prod_prefix}(数字短语)", f"{prod_prefix}(HyDE)", f"{prod_prefix}(全开)"]

    def _print_table(results: dict, title: str, note: str = "") -> None:
        print(f"\n{title}")
        print(f"{'配置':<24}{'NDCG@8':<12}{'Rec@8':<12}{'父块Rec@8':<12}{'MRR':<12}")
        if note:
            print(note)
        print("-" * 62)
        for col in ordered:
            r = results[col]
            print(
                f"{col:<24}{r['ndcg']:<12.4f}{r['rec']:<12.4f}"
                f"{r['prel']:<12.4f}{r['mrr']:<12.4f}"
            )

    _print_table(results, "整体", note=(
        "注：no-rerank 下父块注入在候选池尾部、进不了 top-8，父块Rec@8 恒为 0；"
        "父块入池检查见探针（@候选池口径）。父块Rec@8 在 rerank 评测下才有区分度。"
    ) if args.no_rerank else "")
    if args.split_metric:
        n_metric = sum(1 for it in items if it[2])
        _print_table(results_metric, f"--- 指标题（n={n_metric}） ---")
        _print_table(results_non, f"--- 非指标题（n={len(items) - n_metric}） ---")

    print("\n--- 逐问题 NDCG@8 ---")
    print(f"{'问题':<34}" + "".join(f"{c[:10]:>12}" for c in ordered))
    for idx, (q, rel, _) in enumerate(items):
        row = f"{q[:32]:<34}"
        for col in ordered:
            row += f"{results[col]['per'][idx]:>12.3f}"
        print(row)

    print("\n--- 逐问题 相关父块召回率@8（相关父块>0 的题） ---")
    print(f"{'问题':<34}" + "".join(f"{c[:10]:>12}" for c in ordered))
    for q, rel, _ in items:
        if not rel_parents_by_q.get(q):
            continue
        row = f"{q[:32]:<34}"
        for col in ordered:
            rank_q = ranked[col][q]
            row += f"{_recall_at_k(rank_q, rel_parents_by_q[q], args.top_k):>12.3f}"
        print(row)

    # 写入 eval_results 表
    def _metrics_block(results: dict) -> dict:
        return {
            "ndcg": {c: round(results[c]["ndcg"], 4) for c in ordered},
            "recall": {c: round(results[c]["rec"], 4) for c in ordered},
            "recall1": {c: round(results[c]["rec1"], 4) for c in ordered},
            "precision": {c: round(results[c]["prec"], 4) for c in ordered},
            "mrr": {c: round(results[c]["mrr"], 4) for c in ordered},
            "parent_recall": {c: round(results[c]["prel"], 4) for c in ordered},
        }

    metrics = {
        "top_k": args.top_k,
        "org": args.org,
        "n_questions": len(items),
        "all": _metrics_block(results),
    }
    if args.split_metric:
        metrics["metric"] = _metrics_block(results_metric)
        metrics["non_metric"] = _metrics_block(results_non)
    payload = {
        "per_question_ndcg": {
            c: {q: round(v, 3) for (q, _rel, _m), v in zip(items, results[c]["per"])} for c in ordered
        },
        "per_question_parent_recall": {
            c: {q: round(_recall_at_k(ranked[c][q], rel_parents_by_q.get(q, set()), args.top_k), 3)
                for q, _rel, _m in items} for c in ordered
        },
        "corpus_chunks": len(corpus_ids),
        "golden_questions": len(items),
    }
    try:
        rec = save_eval_result("ablation", f"org={args.org};top_k={args.top_k}", metrics, payload)
        print(f"\n已写入 eval_results: id={rec.id}")
    except Exception as e:
        print(f"写入 eval_results 失败（忽略）: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
