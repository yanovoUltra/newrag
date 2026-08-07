"""元数据层：从文件名与正文推断财报年份-季度；org/权限由上传方显式提供。"""

from __future__ import annotations

import re

_FY_RE = re.compile(r"(20\d{2})")
_QUARTER_RE = re.compile(r"[Qq一二三四1-4]{1}季度?|[Qq][1-4]")


def guess_fiscal_meta(filename: str, sample_text: str = "") -> tuple[int | None, int | None]:
    """返回 (fiscal_year, fiscal_quarter)，均可能为 None。"""
    year = None
    quarter = None

    m = _FY_RE.search(filename)
    if m:
        year = int(m.group(1))

    q = _QUARTER_RE.search(filename)
    if q:
        s = q.group(0).lower()
        if "q" in s:
            quarter = int(s[-1])
        elif "四" in s or "4" in s:
            quarter = 4
        elif "三" in s or "3" in s:
            quarter = 3
        elif "二" in s or "2" in s:
            quarter = 2
        else:
            quarter = 1

    if year is None:
        m = _FY_RE.search(sample_text[:2000])
        if m:
            year = int(m.group(1))
    return year, quarter
