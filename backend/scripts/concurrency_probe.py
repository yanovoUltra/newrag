"""并发验证：同一组问题「串行 vs 并发」问答总耗时对比。

验证目标：阻塞调用线程池化后，单 worker 事件循环不再被嵌入/rerank 同步阻塞占死。
- 串行轮：6 问逐个问 → 总耗时 ≈ 6 × 单问延迟（线性叠加）
- 并发轮：6 问同时发（asyncio.gather）→ 若线程池化生效，总耗时 ≈ 单问延迟附近（非线性）
对比前清空答案缓存桶 + 嵌入缓存（Redis），确保两轮都走全流程、无缓存干扰。
"""

import asyncio
import json
import time

import httpx

BASE = "http://localhost:8000/api/v1"
PAYLOAD = {"org_id": "default", "user_visibility": "public", "session_id": ""}

QUESTIONS = [
    "浦发银行2024年归属于上市公司股东的净利润是多少亿元？",
    "浦发银行2024年末总资产是多少亿元？",
    "浦发银行2024年不良贷款率是多少？",
    "浦发银行2024年基本每股收益是多少元？",
    "苹果公司2024财年毛利率是多少？",
    "苹果公司2024财年净利润是多少？",
]


def flush_redis_cache():
    """清空答案缓存桶与嵌入缓存，确保并发轮无缓存干扰。"""
    try:
        import redis

        r = redis.Redis.from_url("redis://localhost:6379", socket_timeout=2)
        n = 0
        for key in list(r.keys("answer_cache:*")) + list(r.keys("emb:*")):
            r.delete(key)
            n += 1
        print(f"  [清理] 删除缓存键 {n} 个")
    except Exception as e:
        print(f"  [清理] Redis 不可用或失败，跳过（并发轮可能有缓存命中）: {e}")


async def ask(client: httpx.AsyncClient, question: str) -> tuple:
    t0 = time.time()
    n_tokens = 0
    status = "ok"
    try:
        async with client.stream(
            "POST", f"{BASE}/chat", json={**PAYLOAD, "question": question}, timeout=240
        ) as resp:
            resp.raise_for_status()
            cur = None
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    cur = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    ev = json.loads(line[len("data:"):].strip())
                    if cur == "token":
                        n_tokens += len(ev.get("delta", ""))
                    if cur == "done":
                        break
    except Exception as e:
        status = f"ERROR: {type(e).__name__}: {e}"
    return question, time.time() - t0, status, n_tokens


async def serial(client, questions):
    t0 = time.time()
    results = []
    for q in questions:
        results.append(await ask(client, q))
    return results, time.time() - t0


async def concurrent(client, questions):
    t0 = time.time()
    results = await asyncio.gather(*(ask(client, q) for q in questions))
    return results, time.time() - t0


def report(label, results, total):
    print(f"\n=== {label} ===")
    for q, dt, st, nt in results:
        print(f"  {dt:6.1f}s  {st:30s}  token={nt:4d}  {q}")
    print(f"  总耗时: {total:.1f}s  平均每问: {total / len(results):.1f}s")


async def main():
    async with httpx.AsyncClient() as client:
        # 两轮都先清缓存：串行轮走全流程、并发轮不受串行轮缓存干扰
        flush_redis_cache()

        # 串行基线
        results_a, ta = await serial(client, QUESTIONS)
        report("串行（6 问逐个）", results_a, ta)

        # 清缓存后并发
        flush_redis_cache()
        results_b, tb = await concurrent(client, QUESTIONS)
        report("并发（6 问同时发）", results_b, tb)

        print(f"\n结论：并发/串行总耗时 = {tb / ta:.2f}x")
        print("  ~1.0x → 事件循环仍被阻塞串行化（线程池改造无效/未生效）")
        print("  明显 <1.0x → 并发生效（阻塞调用已让出事件循环）")


if __name__ == "__main__":
    asyncio.run(main())
