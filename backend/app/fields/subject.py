"""问题主体公司识别：定位问题所述的所属公司，用于字段抽取的按公司过滤。

背景：字段抽取按 (指标, 年份) 全 org 查询会返回多公司混合值（跨公司噪声）。这里从问题中
抽取出主体公司，再与字段索引中的公司名匹配，从而把字段限定到该公司。

覆盖场景：
- 问题含完整公司名（"贵州茅台2025年…"）→ 直接命中；
- 问题含公司简称（"浦发银行" vs 内容"上海浦东发展银行"）→ 双向子串匹配；
- 问题无公司名（"公司2023年…"）或含多个公司 → 返回 None，调用侧跳过字段注入（避免噪声）。
"""

from __future__ import annotations

import re

from app.fields.metrics import METRIC_CATALOG

# 年份跨度（中文"2024年"/"2024年度"、英文 fiscal year 2024 / FY2024）
_YEAR_SPAN_RE = re.compile(
    r"(?:19|20)\d{2}\s*年度?|fiscal\s+year\s+(?:19|20)\d{2}|FY(?:19|20)\d{2}", re.I
)
_FILLER_RE = re.compile(r"[，,。？?、\s]+")
# 通用填充词（非公司、非指标）
_FILLER_TOKENS = ("请问", "是多少", "为多少", "有多少", "多少", "分别为", "分别", "其", "公司", "的")


def _strip_metric_aliases(q: str) -> str:
    """从问题中剔除命中的指标别名整段（复用指标目录，取最长匹配，避免留下指标词）。"""
    best = ""
    for m in METRIC_CATALOG:
        for alias in m["aliases"]:
            a = alias.lower()
            if a and a in q and len(a) > len(best):
                best = a
    return q.lower().replace(best, "") if best else q


def extract_subject(question: str) -> str:
    """从问题中抽出主体公司候选串：剔除年份/指标别名/填充词后剩余的核心串。"""
    q = _YEAR_SPAN_RE.sub(" ", question or "")
    q = _strip_metric_aliases(q)
    for tok in _FILLER_TOKENS:
        q = q.replace(tok, "")
    q = _FILLER_RE.sub("", q)
    return q


def _normalize_name(name: str) -> str:
    """公司名归一化：繁体→简体（用于招商銀行/招商银行这类差异）。"""
    s = (name or "").strip()
    return s.translate(_TR_TO_SIMPLE)


# 常见繁体字 → 简体（覆盖公司名高频字）
_TR_TO_SIMPLE = str.maketrans({
    "銀": "银", "華": "华", "證": "证", "東": "东", "發": "发",
    "國": "国", "電": "电", "網": "网", "訊": "讯", "數": "数", "據": "据",
    "業": "业", "務": "务", "萬": "万", "億": "亿", "總": "总", "資": "资",
    "產": "产",
})


def _is_subseq(short: str, long: str) -> bool:
    """判断 short 是否为 long 的子序列（保持字符相对顺序）。
    用于简称→全称匹配，如"浦发银行"是"上海浦东发展银行"的子序列（浦→发→银→行）。
    """
    it = iter(long)
    return all(ch in it for ch in short)


def find_subject_company(question: str, companies: list[str]) -> str | None:
    """在公司名列表里定位问题主体公司；0 或 多个匹配返回 None（无法确定 → 跳过字段注入）。

    双向子串 + 双向子序列匹配（归一化简体后，英文统一小写）：
    - 完整名直接命中（"贵州茅台"⊆"贵州茅台"、"CXMT"⊆"CXMT"）；
    - 简称命中全称（"浦发银行"⊆"上海浦东发展银行" 的子序列）；
    - 繁体/简体差异（"招商银行" vs "招商銀行"）；
    - 英文大小写差异（问题提取主体串为小写，公司名保持原样，如 "CXMT"）。
    """
    subj = extract_subject(question)
    if len(subj) < 2:
        return None
    subj_n = _normalize_name(subj)
    matched = [
        c for c in companies
        if c and _match_names(subj_n, _normalize_name(c))
    ]
    if len(matched) == 1:
        return matched[0]
    return None


def _match_names(subj: str, comp: str) -> bool:
    """主体串与公司名的匹配：子串或子序列（双向）。子序列用于简称→全称。

    英文统一转小写再比较（主体提取串来自问题，大小写不保证；公司名保持原始大小写，
    如问题 "What was CXMT's …" → 主体 "whatwascxmt's" vs 公司名 "CXMT"）。
    """
    if not subj or not comp:
        return False
    subj, comp = subj.lower(), comp.lower()
    if subj in comp or comp in subj:
        return True
    return _is_subseq(subj, comp) or _is_subseq(comp, subj)