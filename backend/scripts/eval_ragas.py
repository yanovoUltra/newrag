"""RAGAS 评测流程（自实现，论文指标一致）+ 离线复评（复用 chat_records 历史回答）。

为什么自实现而非 pip ragas：
- 本机 Python 3.13 + pydantic 2.13，ragas 官方依赖链（langchain/datasets/tiktoken）不支持 3.13 且会与项目依赖冲突；
- 评测 LLM 直接复用项目套餐端点（deepseek-v4-flash-0731），零额外配置、自动走限流/重试/缓存；
- 指标定义与 RAGAS 论文一致，可由离线评测结果解释。

指标：
  1. faithfulness（忠实度）：答案拆原子声明 → LLM 逐条判"是否被给定知识块支撑" → 支撑占比
  2. answer_relevancy（答案相关性）：LLM 由答案反生成 3 个问题 → 与原始问题嵌入余弦均值
  3. context_precision（上下文精确率）：LLM 逐知识块判是否相关 → 按原检索顺序 RAGAS 公式

离线复评成本说明（用户目标：减少重复大模型调用成本 ~30%）：
- 历史回答已由 chat_records 持久化，离线复评**不再重新生成回答**（省掉最贵的生成段 LLM 调用）；
- 每问仅 3 次短 prompt 评判调用（声明支撑 1 + 反生成问题 1 + 知识块相关 1），远小于 1 次长流生成；
- contexts 通过 question 重新检索获得（嵌入走缓存，Qdrant/rerank 本地快速）。

用法（backend/ 下）：
    python scripts/eval_ragas.py --mode list                      # 列出可离线复评的历史记录
    python scripts/eval_ragas.py --mode answer --golden            # 对 golden 集生成真实回答并落库（供复评）
    python scripts/eval_ragas.py --mode offline --limit 8          # 离线复评 chat_records 最近 N 条
结果写入 eval_results 表（eval_name=ragas_offline）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.embed.embedder import get_embedder  # noqa: E402
from app.generation.llm import get_llm  # noqa: E402
from app.retrieval.search import hybrid_search  # noqa: E402
from app.store.registry import init_db, list_chat_records, save_eval_result  # noqa: E402

GOLDEN_FILE = SCRIPT_DIR / "eval_golden.json"

_SKIP_PREFIX = ("（Mock", "文档信息不足", "抱歉")


def _clean_answer(a: str) -> str:
    a = (a or "").strip()
    return "" if not a or a.startswith(_SKIP_PREFIX) else a


# ---------------- RAGAS 指标（LLM 评判） ----------------

async def _llm_json(llm, system: str, user: str, attempts: int = 2) -> dict | list | None:
    """带 JSON mode 的 LLM 调用，失败重试。"""
    for _ in range(attempts):
        try:
            text = await llm.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            obj = json.loads(text)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            continue
    return None


async def faithfulness(llm, question: str, answer: str, contexts: list[str]) -> float:
    """RAGAS faithfulness：答案声明中被上下文支撑的比例。"""
    sys_p = (
        "你是评测助手。把用户回答拆分为若干独立原子声明（每个只含一个可验证事实）。"
        "然后逐一判断每个声明是否被给定【知识块】支撑。"
        '只输出 JSON：{"claims":[{"claim":"...","supported":true/false}]}。'
    )
    user_p = (
        f"问题：{question}\n\n答案：{answer}\n\n"
        "【知识块】\n" + "\n---\n".join(c[:800] for c in contexts[:6])
    )
    obj = await _llm_json(llm, sys_p, user_p)
    if not isinstance(obj, dict):
        return 0.0
    claims = obj.get("claims") or []
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if c.get("supported"))
    return supported / len(claims)


async def answer_relevancy(llm, question: str, answer: str) -> float:
    """RAGAS answer_relevancy：由答案反生成 3 个问题，与原问题嵌入余弦均值。"""
    sys_p = "你是评测助手。根据给定答案，生成 3 个该答案可以回答的、独立完整的问题。只输出 JSON：{\"questions\":[...]}。"
    obj = await _llm_json(llm, sys_p, f"答案：{answer[:1200]}")
    if not isinstance(obj, dict):
        return 0.0
    questions = [q for q in (obj.get("questions") or []) if isinstance(q, str) and q.strip()][:3]
    if not questions:
        return 0.0
    try:
        embedder = get_embedder()
        q_list, _ = await asyncio.to_thread(
            embedder.embed_texts_with_sparse, [question] + questions, text_type="query"
        )
        import math

        def _cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(x * x for x in b)) or 1.0
            return dot / (na * nb)

        return sum(_cos(q_list[0], q_list[i]) for i in range(1, len(q_list))) / (len(q_list) - 1)
    except Exception:
        return 0.0


async def context_precision(llm, question: str, contexts: list[str]) -> float:
    """RAGAS context_precision：按检索顺序，位置 i 处精确率（前 i 中相关占比）加权均值。"""
    if not contexts:
        return 0.0
    sys_p = (
        "你是评测助手。判断每个【知识块】是否与给定问题相关（能帮助回答问题）。"
        '只输出 JSON：{"relevant":[true/false, ...]}，顺序与知识块一一对应。'
    )
    user_p = f"问题：{question}\n\n【知识块】\n" + "\n---\n".join(
        f"[{i + 1}] {c[:400]}" for i, c in enumerate(contexts[:8])
    )
    obj = await _llm_json(llm, sys_p, user_p)
    if not isinstance(obj, dict):
        return 0.0
    rel = [bool(x) for x in (obj.get("relevant") or [])][: len(contexts)]
    if not rel:
        return 0.0
    denom = sum(rel)
    if denom == 0:
        return 0.0
    acc = 0.0
    for i, r in enumerate(rel, start=1):
        if r:
            prec = sum(rel[:i]) / i
            acc += prec
    return acc / denom


# ---------------- 主流程 ----------------

def _retrieve_contexts(question: str, top_k: int = 8) -> list[str]:
    """按 question 重新检索知识块（嵌入走缓存），返回内容列表（保持检索顺序）。"""
    settings = get_settings()
    embedder = get_embedder()
    dense, sparse = embedder.embed_texts_with_sparse([question], text_type="query")
    hits = hybrid_search(
        query_vector=dense[0],
        org_id="default",
        user_visibility="public",
        query_text=question,
        top_k=top_k,
        query_sparse=sparse[0] if sparse else None,
    )
    return [h.get("content") or "" for h in hits]


async def _answer_golden() -> int:
    """对 golden 集跑真实问答（无 session_id，走答案缓存 + 落库 chat_records）。"""
    from app.pipelines.answer import stream_answer

    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))["questions"]
    n = 0
    for item in golden:
        q = item["question"]
        answer_text = ""
        async for evt in stream_answer(q, "default", "public", session_id=""):
            if evt["event"] == "token":
                answer_text += evt["data"].get("delta") or ""
        ok = len(_clean_answer(answer_text)) > 20
        print(f"  [{'OK ' if ok else 'SKIP'}] {q[:36]} ({len(answer_text)} 字)")
        if ok:
            n += 1
    print(f"完成，有效回答 {n}/{len(golden)} 条（已落库 chat_records，供离线复评）")
    return n


async def _list_records() -> int:
    recs = [r for r in list_chat_records(limit=100) if _clean_answer(r.answer)]
    print(f"可复评历史记录: {len(recs)} 条")
    for r in recs:
        try:
            n_cit = len(json.loads(r.citations or "[]"))
        except Exception:
            n_cit = 0
        print(f"  {r.created_at:%m-%d %H:%M} | {r.question[:40]} | {len(r.answer)} 字 | citations={n_cit}")
    return len(recs)


async def _offline_eval(limit: int) -> int:
    settings = get_settings()
    llm = get_llm()
    recs = [r for r in list_chat_records(limit=100) if _clean_answer(r.answer)]
    recs = recs[:limit]
    if not recs:
        print("没有可复评的历史记录（先跑 --mode answer 生成）")
        return 1
    print(f"离线复评 {len(recs)} 条 ...")
    rows = []
    for r in recs:
        question, answer = r.question, _clean_answer(r.answer)
        contexts = _retrieve_contexts(question)
        f = await faithfulness(llm, question, answer, contexts)
        rl = await answer_relevancy(llm, question, answer)
        cp = await context_precision(llm, question, contexts)
        rows.append({"question": question, "faithfulness": round(f, 3),
                     "answer_relevancy": round(rl, 3), "context_precision": round(cp, 3),
                     "n_contexts": len(contexts)})
        print(f"  f={f:.3f} rel={rl:.3f} cp={cp:.3f} | {question[:40]}")
    metrics = {
        "faithfulness": round(sum(x["faithfulness"] for x in rows) / len(rows), 4),
        "answer_relevancy": round(sum(x["answer_relevancy"] for x in rows) / len(rows), 4),
        "context_precision": round(sum(x["context_precision"] for x in rows) / len(rows), 4),
        "questions": len(rows),
        "llm_calls_per_question": 3,
        "saved_generation_calls": len(rows),  # 离线复评省掉等量的回答生成调用
    }
    payload = {"rows": rows, "notes": "自实现 RAGAS 指标；contexts 由 question 重新检索（嵌入走缓存）"}
    try:
        rec = save_eval_result("ragas_offline", f"limit={len(rows)};org=default", metrics, payload)
        print(f"\n已写入 eval_results: id={rec.id}")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"写入 eval_results 失败: {e}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS 评测（自实现）+ 离线复评")
    parser.add_argument("--mode", default="list", choices=["list", "answer", "offline"])
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    init_db()
    if args.mode == "list":
        return asyncio.run(_list_records())
    if args.mode == "answer":
        return asyncio.run(_answer_golden())
    if args.mode == "offline":
        return asyncio.run(_offline_eval(args.limit))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
