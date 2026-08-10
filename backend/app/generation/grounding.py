"""接地校验（幻觉防线）：抽取回答中的数字/百分比，回查其是否出现在知识块中。

- 只做确定性校验（归一化：去逗号/空格/全半角 + 数值语义匹配），不调用额外模型；
- 数字未命中等同"该数字在检索到的知识块中找不到"，提示用户核对；
- 同时统计回答中的引用标记数量（[文档-章节-页码] 格式），供前端展示。
"""

from __future__ import annotations

import math
import re

# 数字（含千分位逗号、小数、百分比后缀）；% 兼容全角 ％
_NUM_RE = re.compile(r"[\d][\d,，]*(?:\.\d+)?\s*[%％]?")
# 引用标记：方括号包裹的"来源-章节-页码"片段
_CITE_RE = re.compile(r"\[[^\[\]]{3,}\]")
# "数字 + 单位" 表达式（单位选长优先；数值换算后用于跨单位等价匹配）
_NUM_UNIT_RE = re.compile(
    r"(?P<num>[\d][\d,，]*(?:\.\d+)?)\s*"
    r"(?P<unit>万亿元|亿元|千万元|百万元|万元|元|千元|亿|万|千|billion|million|thousand|B|M|K|k|[$￥¥])?"
)
# 中文/英文单位 → 换算为"元"的倍数
_UNIT_MULT = {
    "万亿元": 1e12, "亿元": 1e8, "千万元": 1e7, "百万元": 1e6, "万元": 1e4,
    "千元": 1e3, "元": 1.0, "亿": 1e8, "万": 1e4, "千": 1e3,
    "billion": 1e9, "million": 1e6, "thousand": 1e3,
    "B": 1e9, "M": 1e6, "K": 1e3, "k": 1e3,
    "$": 1.0, "￥": 1.0, "¥": 1.0,
}


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


def normalize_number_expr(text: str) -> list[tuple[float, str]]:
    """提取文本中"数字+单位"表达式并换算为标准元值。

    返回 [(标准元值, 原始片段)]；无单位/纯货币符号（$ 等）的数字不纳入
    （交由字符串匹配），避免把"2024 年"这类年份/普通数字误当金额。
    """
    out: list[tuple[float, str]] = []
    for m in _NUM_UNIT_RE.finditer(text):
        num_s = m.group("num")
        unit = (m.group("unit") or "").strip()
        raw = m.group(0).strip()
        if not num_s:
            continue
        if not unit or unit in ("$", "￥", "¥"):
            continue  # 无单位不做数值匹配
        try:
            val = float(num_s.replace(",", "").replace("，", ""))
        except ValueError:
            continue
        out.append((val * _UNIT_MULT.get(unit, 1.0), raw))
    return out


def _expr_matched(
    answer_expr: list[tuple[float, str]], corpus_vals: list[float]
) -> set[str]:
    """返回回答中数值语义命中（与 corpus 中任一标准元值近似相等）的原始片段集合。"""
    hit: set[str] = set()
    for v, raw in answer_expr:
        if any(math.isclose(v, cv, rel_tol=1e-3, abs_tol=1.0) for cv in corpus_vals):
            hit.add(raw)
    return hit


def ground_answer(answer: str, blocks: list[dict]) -> dict:
    """校验回答数字是否可在知识块中定位。返回 {checked, missing, citations}。

    blocks 为检索命中（含 content 与可选 parent_content）。
    数字提取前先剔除 [来源-章节-页码] 引用片段，避免文档名/页码误报。
    匹配规则：① 字符串级（归一化后子串）；② 数值语义级（"17.07亿元"≈"170,748万元"）。
    """
    if not answer:
        return {"checked": 0, "missing": [], "citations": 0}
    corpus = "\n".join(
        b.get("content", "") + "\n" + (b.get("parent_content") or "") for b in blocks
    )
    corpus_n = _norm(corpus)
    # 剔除引用片段（如 [600000_2024年报.pdf-第八节 财务报告-第7页]）再抽数字
    body = _CITE_RE.sub("", answer)

    ans_expr = normalize_number_expr(body)
    corpus_vals = [v for v, _ in normalize_number_expr(corpus)]
    expr_hit = _expr_matched(ans_expr, corpus_vals)

    missing: list[str] = []
    for num in extract_numbers(body):
        # % 结尾时同时回查去 % 的数字，容忍原文不带百分号
        bare = _norm(num).rstrip("%")
        if _norm(num) in corpus_n or bare in corpus_n:
            continue
        # 数字属于已数值命中的带单位表达式（如 "1,707,480" 属于 "1,707,480万元"）
        if any(bare in _norm(raw) for raw in expr_hit):
            continue
        missing.append(num)
        if len(missing) >= 10:
            break

    return {
        "checked": len(extract_numbers(body)),
        "missing": missing,
        "citations": len(_CITE_RE.findall(answer)),
    }
