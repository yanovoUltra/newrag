"""字段抽取：从解析结果（版式层，含表格与正文）抽取结构化指标 (year, metric, value)。

抽取出 财务字段 → 独立索引（financial_fields 表），支撑"某年某指标"类问答的精确取值，
弥补检索层在指标级表格问答上的固有短板（检索命中表格行难，结构化取值直接）。

支持两种表格版式：
- 模式 A：指标在行、年份在表头（最常见：项目 | 2024年 | 2023年 | ...）；
- 模式 B：指标在表头、年份在行首（转置表：营业收入 | 净利润：2024年 300 50 ...）。
正文侧做轻量兜底（行内同时含指标别名 + 年份 + 数值）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.logging import get_logger
from app.fields.metrics import (
    MONETARY_METRICS,
    match_metric_aliases,
    parse_number,
    parse_year,
    table_unit,
)
from app.parsers.base import LayoutResult
from app.parsers.structure import section_path_for_page

logger = get_logger(__name__)


@dataclass
class FieldRecord:
    doc_id: str
    metric: str  # canonical key
    metric_label: str
    year: int
    value: float
    unit: str | None = None
    raw: str = ""
    source: str = "table"  # table | text
    page: int = 0
    section_path: str = ""
    company: str = ""  # 所属公司名（内容提取，用于按主体过滤防跨公司噪声）

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "metric": self.metric,
            "metric_label": self.metric_label,
            "year": self.year,
            "value": self.value,
            "unit": self.unit,
            "raw": self.raw,
            "source": self.source,
            "page": self.page,
            "section_path": self.section_path,
            "company": self.company,
        }


# ---- 公司名提取（从文档内容，优先"公司简称X"，其次"XX股份有限公司"类标题） ----
_COMPANY_PATTERNS = [
    re.compile(r"公司简称[:：]?\s*([\u4e00-\u9fffA-Za-z（）()]{2,20})"),
    re.compile(r"([\u4e00-\u9fffA-Za-z]{2,20}?)(?:科技)?股份有限公司"),
    re.compile(r"([\u4e00-\u9fffA-Za-z]{2,20}?)(?:有限|股份)公司"),
    # 英文 10-K/20-F：封面"Apple Inc."/"Tesla, Inc."（须以公司后缀结尾，避免误配机构名）
    re.compile(r"([A-Z][A-Za-z0-9&.' -]{1,40}?)(?:,?\s+(?:Inc\.?|Corp\.?|Corporation|LLC|L\.L\.C\.|Ltd\.?|Limited|Co\.?))"),
]


def extract_company(layout: LayoutResult) -> str:
    """从版式内容提取公司名。文件名常不含公司名（如 600000_2024年报.pdf），故从内容取。

    注意英文 10-K 封面文本长（SEC 封面表格化文本，公司名可能在数万字符之后），
    需扫描整页文本而非仅前 2K 字符（实测 AAPL "Apple Inc." 位于第 14056 字符）。
    """
    for page in layout.pages[:6]:
        text = page.text or ""
        for pat in _COMPANY_PATTERNS:
            m = pat.search(text)
            if m:
                company = m.group(1).strip()
                if company:
                    return company
    return ""


# 比率/百分比类指标：取值应为一个合理的率（≤1000），超大数值几乎都是表格错位误检
_RATIO_METRICS = {"roe", "roa", "debt_ratio", "gross_margin", "net_margin"}


def _plausible_value(key: str, value: float) -> bool:
    """取值合理性守卫：拒绝错位带入的明显异常值。
    - 比率类指标：拒绝 >1000 的巨额数值（如 资产负债率=8,339,591）；
    - 金额类指标：拒绝绝对值 <100 的微小值（营收/资产不可能是 2 元——
      英文表头/数据列错位时会把 Change 列的增幅（2/%）当成指标值）。
    """
    if key in _RATIO_METRICS and value > 1000:
        return False
    if key in MONETARY_METRICS and abs(value) < 100:
        return False
    return True


# ---- 年份表头判定（排除季度/期中/期初列）----
# 允许"2023年 本期/本年/本报告期"这类年报表头（仍是年度列）；
# 排除"2026年1-3月""2026年一季度"等季度/期中表头。
_YEAR_PURE_RE = re.compile(r"^(?:19|20)\d{2}(?:年|年度)?(?:本期|本年|本报告期)?$")
_YEAR_END_RE = re.compile(r"^(?:19|20)\d{2}年12月(?:31|30)?日?$")
# 英文期末日表头："September 28, 2024"（美国/英式财报表头，如 Apple 10-K 资产负债表）
_MONTHS_EN = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_YEAR_END_EN_RE = re.compile(rf"^{_MONTHS_EN}\s+\d{{1,2}},\s*((?:19|20)\d{{2}})$", re.I)
# 增幅/占比列表头关键词（"较上年同期增减"、"Change"、"占比(%)" 等，取值须排除）
_GROWTH_HEADER_RE = re.compile(
    r"同比|增减|变动|增幅|增长率|增长|占比|比重|幅度|change|variance|increase|decrease", re.I
)
# 占比行过滤："Percentage of total net sales" 这类行不是指标值行（数值是占比，非金额）
_PERCENT_OF_LINE_RE = re.compile(r"percentage of total|percent of total|占比", re.I)
# 正文行期初/季度上下文（行内含季度/期中语义时数值多为累计/占比，跳过）
_PERIOD_CTX_RE = re.compile(r"季度|半年度|上半年|期初|(?:[1-9]|1[0-2])月")
# 报告期声明（纯文本财报年份序列来源）："fiscal year ended December 31, 2025" / "Year Ended December 31, 2024"
_FISCAL_YEAR_END_RE = re.compile(
    r"(?:fiscal year|year(?:s)?)\s+ended\s+(?:december|dec\.?)\s+\d{1,2},?\s*(?:19|20)\d{2}",
    re.I,
)


def _is_year_cell(text) -> int | None:
    """单元格是否为年度表头：纯年份（2024 / 2024年 / 2024年度）、期末日（2024年12月31日）
    或英文期末日（September 28, 2024）。

    排除季度/期中/期初表头（如"2026年1-3月""2026年一季度"）——这类列的数值是
    构成占比或累计值，混入会把季度数据当年度（此前误抽 2026 假年份的根因）。
    返回年份 int 或 None。
    """
    raw = (text or "").strip()
    if raw:
        m = _YEAR_END_EN_RE.match(raw)
        if m:
            return int(m.group(1))
    s = raw.replace(" ", "").replace("\u00a0", "")
    m = _YEAR_PURE_RE.match(s)
    if m:
        return int(s[:4])
    m = _YEAR_END_RE.match(s)
    if m:
        return int(s[:4])
    return None


def _year_columns(headers: list[str]) -> dict[int, int]:
    """表头中识别年份列：返回 {列索引: 年份}（仅纯年份/期末日表头）。"""
    mapping: dict[int, int] = {}
    for idx, h in enumerate(headers):
        y = _is_year_cell(h)
        if y is not None:
            mapping[idx] = y
    return mapping


def _reject_percent_for_monetary(key: str, unit: str | None) -> bool:
    """金额类指标抽到百分比（占比/增幅列错位带入）→ 拒绝。如"营业利润 1351.78%"必是增幅列。"""
    return key in MONETARY_METRICS and unit == "%"


def _row_year_values(
    cells: list[str],
    year_cols: dict[int, int],
    headers: list[str],
    label_idx: int,
    key: str = "",
) -> dict[int, tuple[str, float, str | None]]:
    """按行定位各年份列的数值：返回 {year: (raw, value, unit)}。

    取数规则（兼容中英两种版式的行列错位）：
    - 先试年份表头列自身（中文表数据列与表头列对齐，offset=0）；
    - 取不到再向右邻域窗口 [ci+1, ci+6] 探（英文 SEC 表数据行比表头稀疏错位，
      "$"货币符号/空列交错，如 AAPL 10-K "Total net sales" 行 2022 值在表头列后 5 列）；
    - 表头与数据行列数不一致时跳过年份列自身（避免取到错位的前一年数值）；
    - 取值需通过合理性守卫（如金额 <100 多为增幅列错位值），拒绝后继续右探；
    - 已用列不重复取（避免 2023 与 2022 拿到同一数值）；
    - 表头含同比/增减/Change 等词的列跳过（增幅/占比列，非指标值）。
    """
    used: set[int] = set()
    out: dict[int, tuple[str, float, str | None]] = {}
    aligned = len(cells) == len(headers)
    for ci, year in sorted(year_cols.items()):
        if aligned:
            candidates = [ci] + list(range(ci + 1, min(ci + 7, len(cells))))
        else:
            candidates = list(range(ci + 1, min(ci + 7, len(cells))))
        for j in candidates:
            if j == label_idx or j in used:
                continue
            if j < len(headers) and headers[j] and _GROWTH_HEADER_RE.search(str(headers[j])):
                continue
            v, unit = parse_number(cells[j])
            if v is None:
                continue
            if _reject_percent_for_monetary(key, unit) or not _plausible_value(key, v):
                continue  # 错位值（增幅/占比）→ 继续右探下一候选列
            out[year] = (cells[j], v, unit)
            used.add(j)
            break
    return out


def _extract_from_table(doc_id: str, page_no: int, headers: list[str], rows: list[list]) -> list[FieldRecord]:
    """从单张表格抽取字段（模式 A + 模式 B）。"""
    out: list[FieldRecord] = []
    # 整表金额单位（如"单位：人民币百万元"），用于金额类指标补全 unit
    tunit = table_unit(headers)

    def eff_unit(key: str, cell_unit: str | None) -> str | None:
        """金额类指标：表头/单元格单位缺失时按"元"兜底；比率/EPS 类不换算。"""
        if key in MONETARY_METRICS:
            return cell_unit or tunit or "元"
        return cell_unit

    # 模式 A：指标在行、年份在表头（按行独立定位数值列，兼容行列错位）
    year_cols = _year_columns(headers)
    if year_cols and rows:
        for row in rows:
            cells = ["" if c is None else str(c) for c in row]
            label_idx: int | None = None
            matched: dict | None = None
            for ci, cell in enumerate(cells):
                # 占比行（"Percentage of total net sales"）数值是占比非金额，跳过整行
                if _PERCENT_OF_LINE_RE.search(cell):
                    break
                m = match_metric_aliases(cell)
                if m is not None:
                    label_idx, matched = ci, m
                    break
            if matched is None or label_idx is None:
                continue
            for year, (raw, v, unit) in _row_year_values(
                cells, year_cols, headers, label_idx, key=matched["key"]
            ).items():
                out.append(
                    FieldRecord(
                        doc_id=doc_id, metric=matched["key"], metric_label=matched["label"],
                        year=year, value=v, unit=eff_unit(matched["key"], unit), raw=raw, source="table", page=page_no,
                    )
                )

    # 模式 B：指标在表头、年份在行首（转置表）
    if headers and rows:
        header_metrics: dict[int, dict] = {}
        for ci, h in enumerate(headers):
            m = match_metric_aliases(str(h))
            if m is not None:
                header_metrics[ci] = m
        if header_metrics:
            for row in rows:
                cells = ["" if c is None else str(c) for c in row]
                if not cells:
                    continue
                # 行首年份同用年度表头判定（排除"2026年1-3月"这类季度/期中行）
                year = _is_year_cell(cells[0])
                if year is None:
                    continue
                for ci, m in header_metrics.items():
                    if ci >= len(cells):
                        continue
                    v, unit = parse_number(cells[ci])
                    if v is None or _reject_percent_for_monetary(m["key"], unit):
                        continue
                    if not _plausible_value(m["key"], v):
                        continue
                    out.append(
                        FieldRecord(
                            doc_id=doc_id, metric=m["key"], metric_label=m["label"],
                            year=year, value=v, unit=eff_unit(m["key"], unit), raw=cells[ci], source="table", page=page_no,
                        )
                    )
    return out


def _doc_year_sequence(text: str) -> list[int]:
    """从文档文本推断连续报告年份序列（最新在前）。

    纯文本 docx（如 TSLA 10-K）的报表是文本行而非表格对象，年份表头在解析中丢失
    （"Total revenues $ 94,827 $ 97,690 $ 96,773 ..."），但文本含
    "For the fiscal year ended December 31, 2025" 这类报告期声明 → 序列 [2025, 2024, 2023]。
    返回空列表表示无法确定（不启用多值分配，中文文档无此模式天然安全）。
    """
    m = _FISCAL_YEAR_END_RE.search(text or "")
    if not m:
        return []
    y = int(re.search(r"(19|20)\d{2}", m.group(0)).group(0))
    return [y, y - 1, y - 2]


def _collect_report_values(s: str) -> list[tuple[str, float]]:
    """报表文本行多值收集：按出现顺序取"主值"（跳过 % 与括号负数——Change 列特征）。

    如 "Total revenues $ 94,827 $ 97,690 $ 96,773 $ (2,863) (3) % $ 917 1 %"
    → [(94,827), (97,690), (96,773)]（Change 值 (2,863)/(3)/% 被剔除，917 超出主值窗口）。
    """
    vals: list[tuple[str, float]] = []
    for m in re.finditer(r"[-\d,]+(?:\.\d+)?|\([-\d,]+(?:\.\d+)?\)|[\d,]+(?:\.\d+)?\s*%", s):
        token = m.group(0).strip()
        v, unit = parse_number(token)
        if v is None or unit == "%":
            continue
        if token.startswith("("):
            continue
        vals.append((token, v))
    return vals


def _extract_from_text(doc_id: str, page_no: int, text: str, yseq: list[int] | None = None) -> list[FieldRecord]:
    """正文兜底：行内含指标别名 + 年份 + 数值 → 抽取。

    无年份的多值报表行（纯文本 docx）在 yseq 存在时按顺序分配年份（2025/2024/2023）。
    """
    out: list[FieldRecord] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # 期初/季度上下文过滤：行内含"1-3月/季度/上半年"等 → 数值多为累计/占比，跳过
        if _PERIOD_CTX_RE.search(s):
            continue
        m = match_metric_aliases(s)
        if m is None:
            continue
        year = parse_year(s)
        v, unit = parse_number(s)
        # 无年份但含多个数值的报表行：按文档报告年份序列分配（纯文本财报专用）
        if year is None and yseq:
            vals = _collect_report_values(s)
            if len(vals) >= 2:
                for k, (raw, val) in enumerate(vals):
                    if k >= len(yseq):
                        break
                    if not _plausible_value(m["key"], val):
                        continue
                    out.append(
                        FieldRecord(
                            doc_id=doc_id, metric=m["key"], metric_label=m["label"],
                            year=yseq[k], value=val, unit=None, raw=raw, source="text", page=page_no,
                        )
                    )
                continue
        if year is None or v is None:
            continue
        if not _plausible_value(m["key"], v):
            continue
        out.append(
            FieldRecord(
                doc_id=doc_id, metric=m["key"], metric_label=m["label"],
                year=year, value=v, unit=unit, raw=s, source="text", page=page_no,
            )
        )
    return out


def _dedup(records: list[FieldRecord]) -> list[FieldRecord]:
    """同一 (metric, year) 去重：优先生效范围（table > text），保留首个有效值。"""
    seen: dict[tuple[str, int], FieldRecord] = {}
    for r in records:
        key = (r.metric, r.year)
        prev = seen.get(key)
        if prev is None:
            seen[key] = r
            continue
        # 表格优先于文本；同源保留首个
        if r.source == "table" and prev.source != "table":
            seen[key] = r
    return list(seen.values())


def extract_fields(doc_id: str, layout: LayoutResult, sections=None) -> list[FieldRecord]:
    """从版式层抽取全部字段并去重。sections 用于标注章节路径（可选）。"""
    company = extract_company(layout)
    # 纯文本财报的年份序列（如 TSLA 10-K），供正文多值报表行分配年份
    yseq = _doc_year_sequence("\n".join((p.text or "") for p in layout.pages))
    records: list[FieldRecord] = []
    for page in layout.pages:
        path = section_path_for_page(sections, page.page_no) if sections else ""
        for t in page.tables:
            for r in _extract_from_table(doc_id, page.page_no, t.headers, t.rows):
                r.section_path = path
                r.company = company
                records.append(r)
        for r in _extract_from_text(doc_id, page.page_no, page.text, yseq=yseq):
            r.section_path = path
            r.company = company
            records.append(r)
    return _dedup(records)
