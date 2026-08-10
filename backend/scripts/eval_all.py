"""统一评测入口：对抗性 20 题（可选 HyDE 变体）。

golden 已目录化于 scripts/golden/（offline_257.json / adversarial_v2.json），
本脚本调用对抗性评测器，逐段打印结果并落库 eval_results。
注：离线 212 题评测已由 eval_ablation.py（消融网格，含 NDCG/Rec/Prec/MRR）承担，
    eval_offline.py 已删除，故此处不再调用。

用法（backend/ 下）：
    python scripts/eval_all.py [--top-k 10] [--k 5,8,10] [--org default] [--visibility public] [--hyde]

    --hyde  追加对抗性 HyDE 变体（仅 D 类综述型用例先用 LLM 生成假设文档再检索）
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="统一评测入口（adversarial）")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k", default="5,8,10")
    parser.add_argument("--org", default="default")
    parser.add_argument("--visibility", default="public")
    parser.add_argument("--hyde", action="store_true", help="追加对抗性 HyDE 变体（D 类综述型）")
    args = parser.parse_args()

    py = sys.executable
    steps: list[tuple[str, list[str]]] = [
        ("对抗性 20 题 baseline", [
            "eval_adversarial.py", "--top-k", str(args.top_k), "--k", args.k,
            "--org", args.org, "--visibility", args.visibility,
        ]),
    ]
    if args.hyde:
        steps.append(("对抗性 20 题 HyDE（D 类综述型）", [
            "eval_adversarial.py", "--top-k", str(args.top_k), "--k", args.k,
            "--org", args.org, "--visibility", args.visibility, "--hyde",
        ]))

    for name, cmd in steps:
        print(f"\n########## {name} ##########")
        ret = subprocess.run(
            [py, str(SCRIPT_DIR / cmd[0]), *cmd[1:]],
            cwd=SCRIPT_DIR.parent,
        )
        if ret.returncode != 0:
            print(f"[eval_all] {name} 失败（exit={ret.returncode}），终止")
            return ret.returncode

    print("\n全部评测完成，结果均已落库 eval_results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
