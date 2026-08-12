"""阶段五：RAGAS 评估（自实现论文一致指标，生产 rerank 口径）。

四指标（均不依赖参考答案，LLM-as-judge + 确定性计算）：
- faithfulness 忠实度：回答中事实陈述被检索上下文支持的比例
  （judge 一次调用：拆陈述 + 逐条判定，JSON 输出）
- answer_relevancy 回答相关性：由回答生成反向问题 → 与原问题嵌入余弦相似度
- context_precision 上下文精确率：检索 top-k 内 golden 相关块精度曲线（确定性，零 LLM）
- context_recall 上下文召回率：golden 相关块内容的关键陈述被检索上下文覆盖比例
  （judge 一次调用）

流程（每问）：生产检索（_search_plan，含 rerank）→ build_answer_messages 生成（强制主模型
deepseek-v4-flash）→ 3 次短评判调用 + 本地嵌入 1 次。共 ~4 次 LLM 调用/问。

样本：分层抽样 ~100 题（指标 40 + 非指标 60 按题型分层，seed 固定可复现）。

用法（backend/ 下）：
    .venv\\Scripts\\python.exe scripts\\eval_ragas.py [--n-metric 40] [--n-non-metric 60]
    [--top-k 8] [--concurrency 4] [--seed 42] [--dry-run 5]
结果落 eval_results（eval_name=ragas）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.core.config import get_settings  # noqa: E402
from app.embed.embedder import get_embedder  # noqa: E402
from app.generation.llm import get_llm  # noqa: E402
from app.generation.prompts import build_answer_messages  # noqa: E402
from app.pipelines.answer import _extract_year, _search_plan  # noqa: E402
from app.retrieval.router import _is_summary_query, classify_query_type_keyword, RoutePlan  # noqa: E402
from app.store import qdrant as qdrant_store  # noqa: E402
from app.store.registry import init_db, save_eval_result  # noqa: E402

import golden_utils  # noqa: E402

GOLDEN_FILE = SCRIPT_DIR / "golden" / "eval_set.json"

# ---- 非指标题题型分层规则（keyword 近似） ----
_TYPE_RULES = [
    ("趋势", ["趋势", "走势", "逐年", "变化", "变动", "演进", "增长情况", "同期"]),
    ("多跳", ["为什么", "原因", "导致", "影响", "如何", "怎样", "综合", "梳理", "说明"]),
    ("综述", ["总结", "综述", "概述", "归纳", "概括", "整体", "总体"]),
    ("比较", ["对比", "比较", "相比", "差异", "分别", "差距"]),
    ("对抗", ["是否", "核实", "正确", "真的", "属实", "吗", "还是"]),
]


def _subtype(q: str) -> str:
    for name, kws in _TYPE_RULES:
        if any(k in q for k in kws):
            return name
    return "其他"


# ---- LLM-as-judge prompts（短调用，JSON 输出） ----
_JUDGE_SYSTEM = (
    "你是一个严谨的检索问答评估助手。只根据给定的材料判断，"
    "不加入外部知识。只输出合法 JSON，不要输出其他内容。"
)

_FAITHFUL_PROMPT = """判断回答中的事实陈述是否被【知识块】支持。

规则：
- 只提取"信息性事实陈述"（含具体数值、结论、定性判断的断言）；
- 忽略纯格式/元表述：如"根据提供的知识块""来源：""注：""如上所述"等；
- 拆分粒度：每个独立数值断言/事实结论单独成为一条 statement（一句话含多个
  数值就拆成多条），至少 1 条、最多 10 条；
- supported=true 的条件：知识块中存在支持该陈述的信息即可——数值允许等价表述与
  单位换算（如 4.61 与 4.61元/股、2.84 与 $2.84、33,939,755,192.78 元 与 339.4 亿）；
  结论允许同义改写（知识块包含该事实即可，不要求逐字匹配）；
- 若陈述由多个知识块的信息综合而来，各组成部分都能被找到依据也判 supported=true；
- 只有知识块中完全没有依据的信息性陈述才判 supported=false。

【知识块】
{contexts}

【回答】
{answer}

输出 JSON：{{"statements": [{{"claim": "陈述文本", "supported": true/false}}]}}"""

_RELEVANCY_PROMPT = """根据下面的回答，反向生成用户最可能提出的问题。
只输出与原回答对应的问题本身，不要包含解释。

【回答】
{answer}

输出 JSON：{{"question": "生成的问题"}}"""

_RECALL_PROMPT = """判断【参考段落】中的关键陈述是否能在【检索到的知识块】中找到依据。

规则：
- 只提取参考段落中的"信息性事实陈述"（含数值、结论的断言），忽略表述性文字；
- found=true 的条件：检索到的知识块包含支持该陈述的信息即可——数值允许等价表述
  与单位换算（如 4.61 与 4.61元/股、2.84 与 $2.84）；结论允许同义改写；
- 只有检索到的知识块中完全没有依据的陈述才判 found=false。

【检索到的知识块】
{contexts}

【参考段落】
{reference}

输出 JSON：{{"statements": [{{"claim": "陈述文本", "found": true/false}}]}}"""


# ---- 指标计算 ----

def _parse_json_obj(text: str) -> dict:
    t = (text or "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _ctx_texts(hits: list[dict], cap: int = 600, content_cap: int = 2000) -> str:
    """构造评判用上下文文本（content 尽量完整 + 父块截断控制 token）。

    content_cap 默认 2000：faithfulness 评判的 contexts 必须与生成一致——
    指标题答案引用的 relay 表格块可能很长，若截断过短（600）数值行被裁掉，
    judge 会在截断后的 contexts 里找不到答案数值而误判 unsupported。
    """
    parts = []
    for i, b in enumerate(hits, 1):
        content = (b.get("content") or "")[:content_cap]
        parent = (b.get("parent_content") or "")[:cap]
        if parent and b.get("chunk_type") != "section":
            parts.append(f"【块{i}】{content}\n【所属章节】{parent}")
        else:
            parts.append(f"【块{i}】{content}")
    return "\n\n".join(parts)


def _ctx_precision(rel: set[str], hits: list[dict]) -> float:
    """context_precision（RAGAS 公式，确定性）：Σ_k (precision@k × rel_k) / |relevant|。"""
    ranked = [h["chunk_id"] for h in hits[:8]]
    relevant_pos = [i for i, c in enumerate(ranked) if c in rel]
    if not relevant_pos:
        return 0.0
    total = 0.0
    for k in relevant_pos:
        p_at_k = sum(1 for c in ranked[: k + 1] if c in rel) / (k + 1)
        total += p_at_k
    return total / len(relevant_pos)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-12)


# ---- 单题流程 ----

async def _prod_real(
    question: str, org_id: str, user_visibility: str, top_k: int, year: int | None
) -> list[dict]:
    """生产真实路径：_search_plan（含 rerank，阶段五口径经用户许可）。"""
    qtype = classify_query_type_keyword(question)
    plan = RoutePlan(
        intent="factual",
        complexity="simple",
        query_type=qtype,
        needs_hyde=_is_summary_query(question),
    )
    return await _search_plan(plan, question, org_id, user_visibility, top_k, fiscal_year=year)


async def _judge(messages: list[dict[str, str]]) -> str:
    llm = get_llm()
    try:
        return await llm.chat(messages)
    except Exception as e:  # noqa: BLE001
        print(f"  [judge fail] {e}")
        return "{}"


async def _run_one(
    item: dict, org: str, vis: str, top_k: int, fp_index: golden_utils.FingerprintIndex,
    corpus: dict,
) -> dict:
    q = item["q"]
    question = q["question"]
    year = _extract_year(question)
    out: dict = {
        "question": question,
        "type": q.get("type", "metric"),
        "subtype": _subtype(question) if q.get("type", "metric") != "metric" else "metric",
        "year": year,
        "doc_id": q.get("doc_id"),
    }

    hits = await _prod_real(question, org, vis, top_k, year)
    out["n_hits"] = len(hits)
    if not hits:
        out.update({"faithfulness": None, "answer_relevancy": None,
                    "context_precision": 0.0, "context_recall": None,
                    "answer": ""})
        return out
    contexts = _ctx_texts(hits)

    # 1) 生成（强制主模型 deepseek-v4-flash，口径统一）
    msgs = build_answer_messages(question, hits)
    try:
        answer = await get_llm().chat(msgs)
    except Exception as e:  # noqa: BLE001
        answer = ""
        out["gen_error"] = str(e)[:200]
    out["answer"] = (answer or "")[:3000]

    # 2) faithfulness
    if answer:
        raw = await _judge([
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": _FAITHFUL_PROMPT.format(
                contexts=contexts[:20000], answer=answer[:2500])},
        ])
        stmts = _parse_json_obj(raw).get("statements") or []
        if stmts:
            sup = sum(1 for s in stmts if s.get("supported"))
            out["faithfulness"] = round(sup / len(stmts), 4)
            out["faith_n"] = len(stmts)
    # 3) answer_relevancy（反向问题嵌入余弦）
    if answer:
        raw = await _judge([
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": _RELEVANCY_PROMPT.format(answer=answer[:1500])},
        ])
        rev_q = (_parse_json_obj(raw).get("question") or "").strip()
        if rev_q:
            try:
                embedder = get_embedder()
                qv, _ = await asyncio.to_thread(
                    embedder.embed_texts_with_sparse, [question], text_type="query"
                )
                rv, _ = await asyncio.to_thread(
                    embedder.embed_texts_with_sparse, [rev_q], text_type="query"
                )
                out["answer_relevancy"] = round(_cosine(qv[0], rv[0]), 4)
            except Exception:  # noqa: BLE001
                out["answer_relevancy"] = None
    # 4) context_precision（确定性）
    rel = golden_utils.resolve_rel(q, fp_index)
    if q.get("doc_id") and rel:
        rel = {c for c in rel if corpus.get(c, {}).get("doc_id") == q["doc_id"]}
    out["context_precision"] = round(_ctx_precision(rel, hits), 4)
    # 5) context_recall（reference=相关块内容，judge 覆盖判定）
    if rel:
        ref_text = _fetch_ref_text(rel, corpus)
        if ref_text:
            raw = await _judge([
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": _RECALL_PROMPT.format(
                    contexts=contexts[:20000], reference=ref_text[:2500])},
            ])
            stmts = _parse_json_obj(raw).get("statements") or []
            if stmts:
                found = sum(1 for s in stmts if s.get("found"))
                out["context_recall"] = round(found / len(stmts), 4)
                out["recall_n"] = len(stmts)
    return out


def _fetch_ref_text(rel: set[str], corpus: dict) -> str:
    """从 Qdrant 拉相关块内容（最多 3 块，每块截断）。"""
    by_doc: dict[str, list[str]] = {}
    for cid in rel:
        info = corpus.get(cid)
        if info and info.get("doc_id"):
            by_doc.setdefault(info["doc_id"], []).append(cid)
    parts: list[str] = []
    for doc_id, cids in list(by_doc.items())[:2]:
        try:
            payloads = qdrant_store.fetch_payloads(doc_id, cids[:3])
        except Exception:  # noqa: BLE001
            continue
        for cid in cids[:3]:
            pay = payloads.get(cid)
            if pay and pay.get("content"):
                parts.append(f"【参考{cid[:8]}】{pay['content'][:800]}")
    return "\n\n".join(parts)[:3000]


async def main() -> int:
    parser = argparse.ArgumentParser(description="阶段五 RAGAS 四指标评估")
    parser.add_argument("--golden", default=None)
    parser.add_argument("--n-metric", type=int, default=40)
    parser.add_argument("--n-non-metric", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--org", default="default")
    parser.add_argument("--visibility", default="public")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", type=int, default=0, help="只跑前 N 题（调试用）")
    args = parser.parse_args()

    settings = get_settings()
    init_db()
    golden = json.loads((Path(args.golden) if args.golden else GOLDEN_FILE).read_text(encoding="utf-8"))["questions"]

    # 分层抽样（seed 固定可复现）
    rng = random.Random(args.seed)
    metric_all = [q for q in golden if q.get("type", "metric") == "metric"]
    non_all = [q for q in golden if q.get("type", "metric") != "metric"]
    metric_s = rng.sample(metric_all, min(args.n_metric, len(metric_all)))
    buckets: dict[str, list[dict]] = {}
    for q in non_all:
        buckets.setdefault(_subtype(q["question"]), []).append(q)
    non_s: list[dict] = []
    n_left = args.n_non_metric
    for t, qs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        take = max(1, round(args.n_non_metric * len(qs) / max(1, len(non_all)))) if n_left > 0 else 0
        take = min(take, len(qs), n_left)
        non_s.extend(rng.sample(qs, take))
        n_left -= take
    while n_left > 0 and non_all:
        qs = [q for q in non_all if q not in non_s]
        if not qs:
            break
        non_s.append(rng.choice(qs))
        n_left -= 1

    questions = metric_s + non_s
    if args.dry_run:
        questions = questions[: args.dry_run]
    print(f"样本: 指标 {sum(1 for q in questions if q.get('type','metric')=='metric')} + "
          f"非指标 {sum(1 for q in questions if q.get('type','metric')!='metric')} = {len(questions)} 题")

    # 语料 map（chunk_id → doc_id/parent_id）+ 指纹索引
    fp_index = golden_utils.FingerprintIndex.build(args.org, args.visibility)
    corpus: dict[str, dict] = {}
    client = qdrant_store.get_client()
    offset = None
    while True:
        res = client.scroll(
            collection_name=settings.qdrant_collection, limit=1000, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in res[0]:
            pay = p.payload or {}
            corpus[pay["chunk_id"]] = {"doc_id": pay.get("doc_id"), "parent_id": pay.get("parent_id")}
        if res[1] is None:
            break
        offset = res[1]

    items = [{"q": q} for q in questions]
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()

    async def _one(it: dict) -> dict:
        async with sem:
            r = await _run_one(it, args.org, args.visibility, args.top_k, fp_index, corpus)
            return r

    outs = await asyncio.gather(*[_one(it) for it in items])
    print(f"完成 {len(outs)} 题，耗时 {time.perf_counter() - t0:.1f}s\n")

    # ---- 汇总 ----
    def _mean(vals: list[float | None]) -> float | None:
        vs = [v for v in vals if v is not None]
        return round(sum(vs) / len(vs), 4) if vs else None

    all_f = [o.get("faithfulness") for o in outs]
    all_r = [o.get("answer_relevancy") for o in outs]
    all_p = [o.get("context_precision", 0.0) for o in outs]
    all_c = [o.get("context_recall") for o in outs]
    print("=== RAGAS 四指标（全部样本） ===")
    print(f"  faithfulness       {_mean(all_f)}")
    print(f"  answer_relevancy    {_mean(all_r)}")
    print(f"  context_precision   {_mean(all_p)}")
    print(f"  context_recall      {_mean(all_c)}")

    for group, is_m in (("metric", True), ("non_metric", False)):
        sub = [o for o in outs if (o["type"] == "metric") == is_m]
        if not sub:
            continue
        print(f"\n=== {group}（n={len(sub)}） ===")
        print(f"  faithfulness       {_mean([o.get('faithfulness') for o in sub])}")
        print(f"  answer_relevancy    {_mean([o.get('answer_relevancy') for o in sub])}")
        print(f"  context_precision   {_mean([o.get('context_precision', 0.0) for o in sub])}")
        print(f"  context_recall      {_mean([o.get('context_recall') for o in sub])}")

    # 落库
    def _grp_metrics(is_m: bool) -> dict:
        sub = [o for o in outs if (o["type"] == "metric") == is_m]
        return {
            "n": len(sub),
            "faithfulness": _mean([o.get("faithfulness") for o in sub]),
            "answer_relevancy": _mean([o.get("answer_relevancy") for o in sub]),
            "context_precision": _mean([o.get("context_precision", 0.0) for o in sub]),
            "context_recall": _mean([o.get("context_recall") for o in sub]),
        }

    metrics = {
        "n": len(outs), "n_metric": sum(1 for o in outs if o["type"] == "metric"),
        "n_non_metric": sum(1 for o in outs if o["type"] != "metric"),
        "top_k": args.top_k, "seed": args.seed, "rerank": True,
        "faithfulness": _mean(all_f), "answer_relevancy": _mean(all_r),
        "context_precision": _mean(all_p), "context_recall": _mean(all_c),
        "metric": _grp_metrics(True),
        "non_metric": _grp_metrics(False),
    }
    payload = {"per_question": outs}
    try:
        rec = save_eval_result(
            "ragas",
            f"org={args.org};top_k={args.top_k};seed={args.seed};n={len(outs)}",
            metrics, payload,
        )
        print(f"\n已写入 eval_results: id={rec.id}")
    except Exception as e:  # noqa: BLE001
        print(f"写入 eval_results 失败（忽略）: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
