"""多 worker 压测：对比不同 uvicorn worker 数下并发问答的耗时分布，确认单 worker 是否到瓶颈。

用法（在 backend 目录下）：
    python scripts/worker_bench.py --workers 1          # 只测单 worker 基线
    python scripts/worker_bench.py --workers 1,2,4      # 对比 1/2/4 workers（默认）
    python scripts/worker_bench.py --port 8002 --questions 8

设计：
- 对每个 worker 数：清缓存（answer_cache/emb）→ 临时拉起 uvicorn（测试端口，避免与线上 8000 冲突）
  → 并发发问（asyncio.gather，同 concurrency_probe 的问题集）→ 停掉服务 → 记录耗时分布
- 并发下总耗时 ≈ 最慢单问；对比不同 worker 数的总耗时 / P50 / P95 判断瓶颈：
  * 多 worker 无提升（<10%）→ 瓶颈在外部 API（嵌入/LLM/rerank 延迟）而非单 worker 事件循环，无需扩 worker
  * 多 worker 明显提升（>30%）→ 单 worker 已到瓶颈，值得 `--workers N`
- 注意：N 个 worker 各自持有独立 FairSemaphore/线程池，会把外部 API 并发放大 N 倍（可能触发 429→重试→变慢），
  扩 worker 时应按配额同步调低 MAX_*_CONCURRENCY（见 .env）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = [
    "浦发银行2024年归属于上市公司股东的净利润是多少亿元？",
    "浦发银行2024年末总资产是多少亿元？",
    "浦发银行2024年不良贷款率是多少？",
    "浦发银行2024年基本每股收益是多少元？",
    "苹果公司2024财年毛利率是多少？",
    "苹果公司2024财年净利润是多少？",
]
PAYLOAD = {"org_id": "default", "user_visibility": "public", "session_id": ""}


def flush_redis_cache():
    """清空答案缓存桶与嵌入缓存，确保每轮都走全流程（无缓存干扰）。"""
    try:
        import redis

        r = redis.Redis.from_url("redis://localhost:6379", socket_timeout=2)
        n = 0
        for key in list(r.keys("answer_cache:*")) + list(r.keys("emb:*")):
            r.delete(key)
            n += 1
        print(f"  [清理] 删除缓存键 {n} 个")
    except Exception as e:
        print(f"  [清理] Redis 不可用或失败（本轮可能有缓存命中）: {e}")


def start_server(port: int, workers: int) -> subprocess.Popen:
    """临时拉起 uvicorn（测试端口），返回子进程。"""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", str(port),
         "--workers", str(workers), "--log-level", "warning"],
        cwd=str(BACKEND_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return proc


async def wait_ready(port: int, timeout: float = 120.0) -> bool:
    """轮询 /healthz 直到 app=ok。"""
    deadline = time.time() + timeout
    async with httpx.AsyncClient(timeout=5) as client:
        while time.time() < deadline:
            try:
                r = await client.get(f"http://127.0.0.1:{port}/healthz")
                if r.status_code == 200 and r.json().get("app") == "ok":
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
    return False


async def ask(client: httpx.AsyncClient, port: int, question: str) -> tuple:
    t0 = time.time()
    n_tokens = 0
    status = "ok"
    try:
        async with client.stream(
            "POST", f"http://127.0.0.1:{port}/api/v1/chat",
            json={**PAYLOAD, "question": question}, timeout=300,
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


async def run_concurrent(port: int, questions: list[str]) -> tuple[list[tuple], float]:
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(ask(client, port, q) for q in questions))
    return results, time.time() - t0


def summarize(label: str, results: list[tuple], total: float) -> dict:
    dts = sorted(r[1] for r in results)
    errors = [r for r in results if not r[2].startswith("ok")]
    n = len(dts)
    stats = {
        "label": label,
        "total": total,
        "avg": total / n,
        "p50": dts[max(0, n // 2 - 1)],
        "p95": dts[min(n - 1, int(n * 0.95))],
        "max": dts[-1],
        "errors": len(errors),
        "throughput": n / total,
    }
    print(f"\n=== {label} ===")
    for q, dt, st, nt in results:
        print(f"  {dt:6.1f}s  {st:50s}  token={nt:4d}  {q}")
    print(
        f"  总耗时={total:.1f}s 平均={stats['avg']:.1f}s "
        f"P50={stats['p50']:.1f}s P95={stats['p95']:.1f}s max={stats['max']:.1f}s "
        f"错误={len(errors)} 吞吐={stats['throughput']:.3f} 问/s"
    )
    return stats


async def bench_one(port: int, workers: int, questions: list[str], keep: bool) -> dict | None:
    print(f"\n########## workers={workers} @ port {port} ##########")
    flush_redis_cache()
    proc = start_server(port, workers)
    try:
        if not await wait_ready(port):
            print("  服务启动超时，跳过该配置")
            return None
        print(f"  服务就绪（healthz ok），开始并发 {len(questions)} 问…")
        results, total = await run_concurrent(port, questions)
        return summarize(f"workers={workers}（并发 {len(questions)} 问）", results, total)
    finally:
        if not keep:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("  服务已停止")


def main():
    ap = argparse.ArgumentParser(description="多 worker 压测对比")
    ap.add_argument("--workers", default="1,2,4", help="要测的 worker 数，逗号分隔（默认 1,2,4）")
    ap.add_argument("--port", type=int, default=8001, help="测试端口（默认 8001，避开线上 8000）")
    ap.add_argument("--questions", type=int, default=6, help="并发问题数（默认 6，取问题集前 N 个）")
    ap.add_argument("--keep", action="store_true", help="测试结束后保留服务（调试用）")
    args = ap.parse_args()

    workers_list = [int(w) for w in args.workers.split(",") if w.strip()]
    questions = DEFAULT_QUESTIONS[: args.questions]

    rows = []
    for w in workers_list:
        stats = asyncio.run(bench_one(args.port, w, questions, args.keep))
        if stats:
            rows.append(stats)
        args.port += 1  # 每个配置独立端口，避免 TIME_WAIT 复用问题

    if len(rows) >= 2:
        print("\n########## 汇总对比 ##########")
        base = rows[0]
        print(f"{'workers':<10}{'总耗时s':<10}{'平均s':<8}{'P95s':<8}{'吞吐(问/s)':<12}{'相对1w'}")
        for r in rows:
            rel = r["total"] / base["total"] if base["total"] else float("inf")
            print(
                f"{r['label'].split('=')[1].split('（')[0]:<10}"
                f"{r['total']:<10.1f}{r['avg']:<8.1f}{r['p95']:<8.1f}"
                f"{r['throughput']:<12.3f}{rel:.2f}x"
            )
        print("\n结论判读（对比 1 worker）：")
        print("  相对 1w >= 0.90（多 worker 几乎无提升）→ 瓶颈在外部 API 延迟，单 worker 未到瓶颈，无需扩 worker")
        print("  相对 1w <= 0.70（明显提速）→ 单 worker 事件循环/CPU 已是瓶颈，值得 --workers N（注意同步调低 MAX_*_CONCURRENCY）")


if __name__ == "__main__":
    main()
