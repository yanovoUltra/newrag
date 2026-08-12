"""数值范围约束：解析"指标+比较符+阈值"三元组，候选块同族数值满足度打分。

支撑"不良贷款率低于1%"、"营收超千亿"类条件查询——语义/字面检索只能匹配
"指标名+数值"，无法校验 >/< 逻辑关系。约束解析后对候选块抽取同族数值验证
满足度，作为奖励分注入最终打分（final = base + β·NumMatchScore，见 search.py）。

阈值支持阿拉伯数字+单位与中文数量词（"千亿"→1000 亿）；单位按族归一
（金额统一到"元"、比率到"百分点"，1 个百分点 = 1%）。
"""

from __future__ import annotations

import re

from app.fields.metrics import extract_metric_from_question

# 比较符 → 算子（长模式在前，防 "不低于" 被 "低于" 提前命中）
_OPERATORS = {
    "不超过": "<=", "不高于": "<=", "不大于": "<=",
    "不低于": ">=", "不少于": ">=", "不小于": ">=",
    "低于": "<", "小于": "<", "少于": "<", "不足": "<",
    "高于": ">", "大于": ">", "超过": ">", "超出": ">", "多于": ">", "超": ">",
    "达到": ">=", "至少": ">=", "等于": "=",
    "至多": "<=", "最多": "<=",
}
_COMPARATOR_RE = re.compile("|".join(map(re.escape, sorted(_OPERATORS, key=len, reverse=True))))

# 阈值：阿拉伯数字+单位 或 中文数量词
_THRESHOLD_ARAB_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(万亿|百亿|十亿|千亿|百万元|千万元|亿元|万元|千元|元|个百分点|%|亿|万|倍)?"
)
_THRESHOLD_CN_RE = re.compile(r"([零一二两三四五六七八九十百千万亿]+)\s*(万亿|亿|万|千|百)?")

_CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}

# 候选块值短语（阿拉伯数字+单位）
_VALUE_RE = re.compile(
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(万亿|百亿|十亿|千亿|亿元|百万元|千万元|万元|千元|元|个百分点|%|亿|万|倍)?"
)


def _cn_to_num(s: str) -> float:
    total, section, num = 0.0, 0.0, 0
    for ch in s:
        if ch in _CN_DIGIT:
            num = _CN_DIGIT[ch]
        elif ch in ("十", "百", "千"):
            section += (num or 1) * _CN_UNIT[ch]
            num = 0
        elif ch in ("万", "亿"):
            section += num
            total += (section or 1) * _CN_UNIT[ch]
            section, num = 0, 0
    return total + section + num


def _to_family(num: float, unit: str) -> tuple[str, float] | None:
    """单位 → (数值族, 归一值)：金额统一到"元"，比率到"百分点"，倍为比值族。"""
    u = unit or ""
    if u in ("%", "个百分点"):
        return "percent", num
    if u == "倍":
        return "ratio", num
    mult = {
        "万亿": 1e12, "千亿": 1e11, "百亿": 1e10, "十亿": 1e9, "亿": 1e8,
        "百万元": 1e6, "千万元": 1e7, "万元": 1e4, "千元": 1e3, "元": 1.0,
        "万": 1e4, "千": 1e3, "百": 100, "": 1.0,
    }.get(u)
    if mult is None:
        return None
    return "money", num * mult


def _parse_threshold(after: str) -> tuple[str, float] | None:
    m = _THRESHOLD_ARAB_RE.match(after)
    if m:
        num = float(m.group(1).replace(",", ""))
        fam, val = _to_family(num, m.group(2) or "")
        return (fam, val) if fam else None
    m = _THRESHOLD_CN_RE.match(after)
    if m:
        fam, val = _to_family(_cn_to_num(m.group(1)), m.group(2) or "")
        return (fam, val) if fam else None
    return None


def parse_numeric_constraints(question: str) -> list[dict]:
    """解析"指标+比较符+阈值"三元组：[{metric, op, value, family}]。无指标/无数值返回 []。"""
    key, _ = extract_metric_from_question(question)
    if not key:
        return []
    cons: list[dict] = []
    for m in _COMPARATOR_RE.finditer(question or ""):
        after = (question or "")[m.end():]
        parsed = _parse_threshold(after)
        if parsed:
            fam, val = parsed
            cons.append({"metric": key, "op": _OPERATORS[m.group(0)], "value": val, "family": fam})
    seen: set[tuple] = set()
    out: list[dict] = []
    for c in cons:
        sig = (c["metric"], c["op"], c["value"], c["family"])
        if sig not in seen:
            seen.add(sig)
            out.append(c)
    return out


def _cmp(v: float, op: str, t: float) -> bool:
    if op == "<":
        return v < t
    if op == "<=":
        return v <= t
    if op == ">":
        return v > t
    if op == ">=":
        return v >= t
    if op == "=":
        return abs(v - t) < 1e-9
    return False


def numeric_match_score(content: str, constraints: list[dict]) -> float:
    """候选块满足度：任一约束被块内同族数值满足计 1，返回满足比例 0~1。"""
    if not constraints:
        return 0.0
    vals_by_family: dict[str, list[float]] = {"money": [], "percent": [], "ratio": []}
    for m in _VALUE_RE.finditer(content or ""):
        num = float(m.group(1).replace(",", ""))
        fam_val = _to_family(num, m.group(2) or "")
        if fam_val:
            fam, val = fam_val
            vals_by_family[fam].append(val)
    hit = 0
    for c in constraints:
        if any(_cmp(v, c["op"], c["value"]) for v in vals_by_family.get(c["family"], [])):
            hit += 1
    return hit / len(constraints)
