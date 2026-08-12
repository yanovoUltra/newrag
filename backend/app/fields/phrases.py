"""数字+单位短语：提取与归一化（离线倒排索引构建侧与查询侧共用同一口径）。

背景：通用 embedding 把数字当作普通 token，对精确数值不敏感（"1.2%" 与 "1.23%"
无法区分）；数字+单位短语独立精确匹配可锚定含该短语的 chunk。
离线构建：scripts/build_number_phrase_index.py（短语 → chunk_id 倒排入库）；
查询侧：hybrid_search 提取 query 短语 → 倒排精确匹配 → 硬插候选集。
"""

from __future__ import annotations

import re

# 单位白名单（长单位在前，避免 "1.2%年" 之类歧义与子串误配）：
# - 金额/比率：% 个百分点 亿元 百万元 千万元 万元 千元 元 万亿 百万 倍
# - 计数量词（网点/设备/人员/笔数等）：万户 万人 家 台 个 户 笔 人 名 次 条 项 处 件
# 4 位年份天然排除（"2024" 后无单位不匹配）；"年/月/日" 不列入避免年份/日期误配。
_UNIT_TOKENS = (
    "个百分点", "百万元", "千万元", "亿元", "千元", "万元", "万亿", "百万", "倍",
    "%",
    "万户", "万人", "家", "台", "个", "户", "笔", "人", "名", "次", "条", "项", "处", "件",
)

_NUMBER_UNIT_RE = re.compile(
    r"(?<![\d,.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{1,3}(?:\.\d+)?)\s*"
    r"(?:" + "|".join(map(re.escape, _UNIT_TOKENS)) + ")"
)

# 短语最长保留字符数（两侧一致截断，保证精确匹配口径；超长短语罕见）
_MAX_PHRASE_LEN = 24


def extract_number_phrases(text: str) -> list[str]:
    """提取"数值+单位"短语（去重、保序、超长截断）。

    例："网点 15,434 家，不良率 1.2%" → ["15,434 家", "1.2%"]。
    区间值（1.5%~2.0%）两端各成一个短语。
    """
    phrases: list[str] = []
    seen: set[str] = set()
    for m in _NUMBER_UNIT_RE.finditer(text or ""):
        p = (m.group(0) or "").strip()
        if not p:
            continue
        if len(p) > _MAX_PHRASE_LEN:
            p = p[: _MAX_PHRASE_LEN]
        if p not in seen:
            seen.add(p)
            phrases.append(p)
    return phrases


def norm_number_phrase(phrase: str) -> str:
    """归一化：去千分位/中英文逗号/空白/不换行空格。

    查询侧与 chunk 侧都必须经过本函数，保证 "15,434 家" 与 "15434家" 精确相等。
    """
    return (
        (phrase or "")
        .replace(",", "")
        .replace("，", "")
        .replace(" ", "")
        .replace("\u00a0", "")
        .replace("\n", "")
    )
