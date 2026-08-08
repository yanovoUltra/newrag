"""数据清洗：页眉/页脚去重 + NFKC 归一化。

- 页眉/页脚去重：跨页高频出现的短行（出现页数占比 >= min_ratio 且 token <= max_tokens），
  且位于页首/页尾边缘区（margin）时剔除；出现在页中部的高频行（如表格内重复表头）不受影响。
- NFKC 归一化：全角数字/逗号/百分号 → 半角，统一数字表达，利于检索与接地校验匹配。
- 全程幂等：对同一 layout 重复执行结果不变，可安全接入断点续跑管线。
"""

from __future__ import annotations

import unicodedata

from app.core.config import get_settings
from app.core.logging import get_logger
from app.parsers.base import LayoutResult
from app.splitter.tokens import count_tokens

logger = get_logger(__name__)


def normalize_text(text: str) -> str:
    """NFKC 归一化：全角字母数字/标点 → 半角，兼容全角百分号/逗号/破折号。"""
    return unicodedata.normalize("NFKC", text)


def clean_layout(layout: LayoutResult, settings=None) -> LayoutResult:
    """就地清洗 layout：NFKC 归一化 + 页眉页脚去重。返回同一对象。"""
    settings = settings or get_settings()
    if not settings.clean_enable:
        return layout

    if settings.clean_nfkc:
        for p in layout.pages:
            p.text = normalize_text(p.text)
            for t in p.tables:
                t.headers = [normalize_text(h) for h in t.headers]
                t.rows = [[normalize_text(c) for c in row] for row in t.rows]

    if settings.clean_header_footer:
        _strip_headers_footers(layout.pages, settings)

    return layout


def _strip_headers_footers(pages, settings) -> None:
    """跨页高频 + 短行 + 页首/页尾边缘区 → 剔除。"""
    n = len(pages)
    if n < 3:
        return

    # 1) 统计每行内容出现的页数（同页去重）
    counts: dict[str, int] = {}
    for p in pages:
        seen: set[str] = set()
        for line in p.text.split("\n"):
            s = line.strip()
            if s:
                seen.add(s)
        for s in seen:
            counts[s] = counts.get(s, 0) + 1

    min_pages = max(2, int(n * settings.clean_header_footer_min_ratio))
    max_tok = settings.clean_header_footer_max_tokens
    candidates = {s for s, c in counts.items() if c >= min_pages and count_tokens(s) <= max_tok}
    if not candidates:
        return

    margin = settings.clean_header_footer_margin
    removed = 0
    for p in pages:
        lines = p.text.split("\n")
        total = len(lines)
        if total == 0:
            continue
        keep: list[str] = []
        for i, line in enumerate(lines):
            s = line.strip()
            pos = i / total
            if s in candidates and (pos <= margin or pos >= 1 - margin):
                removed += 1
                continue
            keep.append(line)
        p.text = "\n".join(keep)
    if removed:
        logger.info("clean: removed %d header/footer lines (min_pages=%d)", removed, min_pages)
