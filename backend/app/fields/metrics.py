"""指标目录与数值/年份解析工具。

字段抽取的对象是"结构化指标"（净利润/每股收益/ROE 等），问题多为"某年某指标是多少"。
本模块提供：
- 指标目录（canonical key / 展示标签 / 别名），供抽取与问答侧匹配；
- 数值解析（千分位/百分号/括号负数/单位），年份解析；
- 问答侧指标与年份提取（从用户问题定位要查的指标与年份）。
"""

from __future__ import annotations

import re

# ---- 指标目录 ----
# 每个指标：key（稳定标识）、label（展示名）、aliases（表格/文本中出现的写法，按特异度从高到低）
# 中英双语别名：中文 A 股财报 + 英文 10-K/20-F（SEC 财报）都覆盖。
METRIC_CATALOG: list[dict] = [
    {"key": "revenue", "label": "营业收入",
     "aliases": ["营业总收入", "营业收入合计", "营业收入", "主营收入", "营收",
                 # 英文仅保留"总计类"强别名：裸 revenue/revenues 会子串误配
                 # "Deferred revenue"（资产负债表递延收入，非营业收入）、"Revenue Recognition" 标题等
                 "total net sales", "net sales", "total revenues"]},
    {"key": "net_profit", "label": "净利润",
     "aliases": ["归属于母公司股东的净利润", "归母净利润", "归属于母公司所有者的净利润",
                 "本公司股东应占溢利", "净利润", "净利",
                 "net income attributable to common shareholders",
                 "net income attributable to shareholders", "net earnings", "net income"]},
    {"key": "eps", "label": "每股收益",
     "aliases": ["基本每股收益", "摊薄每股收益", "稀释每股收益", "每股收益", "每股盈余", "eps",
                 "basic earnings per share", "diluted earnings per share",
                 "basic net income per share", "earnings per share"]},
    {"key": "roe", "label": "净资产收益率",
     "aliases": ["加权平均净资产收益率", "净资产收益率", "roaa", "roe", "return on equity"]},
    {"key": "roa", "label": "总资产收益率",
     "aliases": ["总资产报酬率", "总资产收益率", "roa", "return on assets"]},
    {"key": "gross_margin", "label": "毛利率",
     "aliases": ["综合毛利率", "销售毛利率", "毛利率", "gross margin"]},
    {"key": "net_margin", "label": "净利率",
     "aliases": ["销售净利率", "净利率", "net margin", "net profit margin"]},
    {"key": "debt_ratio", "label": "资产负债率",
     "aliases": ["资产负债率", "负债率", "debt-to-assets ratio", "debt to assets ratio",
                 "total debt to total assets"]},
    {"key": "operating_cashflow", "label": "经营活动现金流量净额",
     "aliases": ["经营活动产生的现金流量净额", "经营活动现金流量净额", "经营现金流净额",
                 "经营现金流", "经营活动净现金流",
                 "net cash provided by operating activities",
                 "net cash from operating activities", "cash provided by operating activities"]},
    {"key": "total_assets", "label": "总资产",
     "aliases": ["资产总计", "总资产", "资产总额", "total assets"]},
    {"key": "net_assets", "label": "净资产",
     "aliases": ["股东权益合计", "所有者权益合计", "归属于母公司股东权益", "净资产",
                 "total shareholders' equity", "total stockholders' equity",
                 "shareholders' equity", "stockholders' equity"]},
    {"key": "gross_profit", "label": "营业利润",
     "aliases": ["营业利润", "operating income", "operating profit", "income from operations"]},
    {"key": "total_profit", "label": "利润总额",
     "aliases": ["利润总额",
                 "total profit", "income before income taxes",
                 "income before provision for income taxes",
                 "income before taxes", "pretax income", "pre-tax income"]},
]

# key → 指标对象
_METRIC_BY_KEY = {m["key"]: m for m in METRIC_CATALOG}


def metric_by_key(key: str) -> dict | None:
    return _METRIC_BY_KEY.get(key)


def metric_label(key: str) -> str:
    m = _METRIC_BY_KEY.get(key)
    return m["label"] if m else key


def match_metric_aliases(text: str) -> dict | None:
    """在文本/单元格中匹配指标别名；返回命中指标（按别名特异度优先）。"""
    t = (text or "").strip().lower()
    if not t:
        return None
    best: tuple[int, dict, str] | None = None
    for m in METRIC_CATALOG:
        for alias in m["aliases"]:
            a = alias.lower()
            if a and a in t:
                # 匹配到越长的别名越特异
                if best is None or len(a) > best[0]:
                    best = (len(a), m, alias)
    return best[1] if best else None


# ---- 金额/比率识别与单位折算 ----

# 金额类指标：单位随报表而定（元/万元/百万元/亿元），需折算到统一基准才可比；
# 比率/每股收益类指标（ROE/ROA/毛利率/EPS 等）本身已是可读标度，不参与单位换算。
MONETARY_METRICS = {
    "revenue", "net_profit", "gross_profit", "total_profit",
    "operating_cashflow", "total_assets", "net_assets",
}

# 单位 → 相对"元"的倍数（未知/None 视为"元"）
_UNIT_TO_YUAN = {
    "元": 1.0, "千元": 1e3, "万元": 1e4, "百万元": 1e6, "百万": 1e6,
    "亿元": 1e8, "十亿元": 1e9, "万亿元": 1e12,
}


def unit_to_yuan(value: float | None, unit: str | None) -> float | None:
    """把某单位下的数值折算为"元"；unit 为空或未知视为"元"（×1）。"""
    if value is None:
        return None
    return value * _UNIT_TO_YUAN.get((unit or "元").strip(), 1.0)


# 表头整表金额单位识别："单位：人民币百万元" / "金额单位：万元" / "单位：千元" 等
_TABLE_UNIT_RE = re.compile(r"(十亿元|百万元|千元|万元|亿元|百万|元)")


def table_unit(headers) -> str | None:
    """从表头单元格识别整表金额单位；找不到返回 None（调用方默认按"元"处理）。"""
    for h in headers or []:
        s = str(h or "")
        if "单位" in s and ("元" in s or "百万" in s):
            m = _TABLE_UNIT_RE.search(s)
            if m:
                return m.group(1)
    return None


# ---- 数值解析 ----

# 单位后缀（含 %），解析后单独返回
_UNITS = ("亿元", "百万元", "万元", "亿元", "元", "%", "百万", "万")
_EMPTY_TOKENS = {"", "-", "—", "－", "--", "不适用", "n/a", "na", "无", "—"}
_YEAR_RE = re.compile(r"(19|20)\d{2}")


def parse_number(text) -> tuple[float | None, str | None]:
    """解析单元格/文本中的数值。返回 (value, unit)；无法解析或为空返回 (None, unit)。

    - 支持千分位逗号、空格、括号负数 (1,234)、百分号、中文单位（亿元/万元）；
    - "—"/"-"/"不适用" 等视为无值。
    """
    s = (text or "").strip()
    if not s:
        return None, None
    low = s.lower()
    if low in _EMPTY_TOKENS:
        return None, None
    s = s.replace("\u00a0", " ").replace(" ", "")
    # 括号负数：(1,234.56) → -1234.56
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    # 提取单位
    unit: str | None = None
    for u in _UNITS:
        if s.endswith(u):
            unit = u
            s = s[: -len(u)]
            break
    # 去掉千分位与约等于/区间符号
    cleaned = s.replace(",", "").replace("~", "").replace("≈", "").replace("约", "")
    try:
        v = float(cleaned)
    except ValueError:
        return None, unit
    # 数值为 0 也有效；但非有限数视为无效
    if v != v or v in (float("inf"), float("-inf")):
        return None, unit
    return v, unit


def parse_year(text) -> int | None:
    m = _YEAR_RE.search(text or "")
    return int(m.group(0)) if m else None


# ---- 问答侧：从问题提取 (指标 key, 年份) ----

def _question_year(question: str, default_year: int | None = None) -> int | None:
    return parse_year(question) or default_year


def extract_metric_from_question(question: str) -> tuple[str | None, int | None]:
    """从问题中提取 (指标 key, 年份)。年份缺失时返回 None（不限定年份）。"""
    q = question or ""
    m = match_metric_aliases(q)
    key = m["key"] if m else None
    year = parse_year(q)
    return key, year