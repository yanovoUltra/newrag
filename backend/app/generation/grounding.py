"""接地校验（幻觉防线）：抽取回答中的数字/百分比，回查其是否出现在知识块中。

- 只做确定性字符串校验（归一化：去逗号/空格/全半角），不调用额外模型；
- 数字未命中等同"该数字在检索到的知识块中找不到"，提示用户核对；
- 同时统计回答中的引用标记数量（[文档-章节-页码] 格式），供前端展示。
"""

from __future__ import annotations

import re

# 数字（含千分位逗号、小数、百分比后缀）；% 兼容全角 ％
_NUM_RE = re.compile(r"[\d][\d,，]*(?:\.\d+)?\s*[%％]?")
# 引用标记：方括号包裹的"来源-章节-页码"片段
_CITE_RE = re.compile(r"\[[^\[\]]{3,}\]")


def _norm(text: str) -> str:
    """归一化：去逗号/空格/不换行空格，用于数字与文本匹配。"""
    return (
        text.replace(",", "")
        .replace("，", "")
        .replace(" ", "")
        .replace("\u00a0", "")
        .replace("％", "%")
    )


def extract_numbers(text: str) -> list[str]:
    """提取回答中的数字/百分比 token（去重保序）。"""
    seen: set[str] = set()
    out: list[str] = []
    for m in _NUM_RE.finditer(text):
        tok = m.group(0).strip()
        if not tok or not any(ch.isdigit() for ch in tok):
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def ground_answer(answer: str, blocks: list[dict]) -> dict:
    """校验回答数字是否可在知识块中定位。返回 {checked, missing, citations}。

    blocks 为检索命中（含 content 与可选 parent_content）。
    数字提取前先剔除 [来源-章节-页码] 引用片段，避免文档名/页码误报。
    """
    if not answer:
        return {"checked": 0, "missing": [], "citations": 0}
    corpus = "\n".join(
        b.get("content", "") + "\n" + (b.get("parent_content") or "") for b in blocks
    )
    corpus_n = _norm(corpus)
    # 剔除引用片段（如 [600000_2024年报.pdf-第八节 财务报告-第7页]）再抽数字
    body = _CITE_RE.sub("", answer)

    missing: list[str] = []
    for num in extract_numbers(body):
        # % 结尾时同时回查去 % 的数字，容忍原文不带百分号
        bare = _norm(num).rstrip("%")
        if _norm(num) in corpus_n or bare in corpus_n:
            continue
        missing.append(num)
        if len(missing) >= 10:
            break

    return {
        "checked": len(extract_numbers(body)),
        "missing": missing,
        "citations": len(_CITE_RE.findall(answer)),
    }
