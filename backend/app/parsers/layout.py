"""版式层：PyMuPDF / PDFPlumber 提取文本与表格。"""

from __future__ import annotations

from pathlib import Path

from app.core.logging import get_logger
from app.parsers.base import DocumentElement, LayoutResult, ParsedPage, ParsedTable

logger = get_logger(__name__)

# 页面上少于该字符数视为扫描页（需 OCR）
SCAN_PAGE_CHARS = 20


def parse_pdf(path: Path) -> LayoutResult:
    import fitz  # PyMuPDF
    import pdfplumber

    pages: list[ParsedPage] = []
    with fitz.open(path) as doc:
        with pdfplumber.open(path) as pdf:
            for page_no, (fz_page, pl_page) in enumerate(zip(doc, pdf.pages), start=1):
                tables: list[ParsedTable] = []
                table_elements: list[DocumentElement] = []
                table_bboxes: list[tuple[float, float, float, float]] = []
                try:
                    found = pl_page.find_tables() or []
                    table_bboxes = [t.bbox for t in found]
                    # 无框线财报主表（浦发/招商/平安等）line 策略检测不到行线，
                    # 当本页没有"有效"表格（≥3列且表头含年份）时，用 text 策略抢救。
                    if not any(_is_good_table(t) for t in found):
                        try:
                            text_tables = pl_page.find_tables(
                                {"vertical_strategy": "text", "horizontal_strategy": "text"}
                            ) or []
                        except Exception:
                            text_tables = []
                        # P3.1：word-atomic 重建需要 page 的 selectable words，
                        # 只在确有 rescue 候选时取一次，避免给普通页增加开销。
                        page_words: list[dict] = []
                        if text_tables:
                            try:
                                page_words = pl_page.extract_words() or []
                            except Exception:
                                page_words = []
                        for t in text_tables:
                            # 只保留"有效"财报表（≥3列且表头含年份）；text 策略会把普通正文/整列数值
                            # 挤成单格垃圾表，若加入会屏蔽正文文本并污染结构化表，故丢弃。
                            # P3.1：先用 word-atomic 重建 cells（word 不再被竖线切开），
                            # 拿不到几何证据时回退原 pdfplumber rows，行为与 A0 一致。
                            rebuilt = _rebuild_rows_word_atomic(t, page_words)
                            if rebuilt is None:
                                if not _is_good_table(t):
                                    continue
                                candidate = t
                            else:
                                if not _is_good_rows(rebuilt):
                                    continue
                                candidate = _WordAtomicTable(t, rebuilt)
                            if any(_bbox_overlap(candidate.bbox, b) for b in table_bboxes):
                                continue
                            found.append(candidate)
                            table_bboxes.append(candidate.bbox)
                    for table in found:
                        raw = table.extract() or []
                        if getattr(table, "word_atomic", False):
                            # P3.1 已保证 word 不被切开，P1 的数值切分修复冗余，
                            # 且其按 table.rows 下标定位，与重建后的行数不一致。
                            cell_repairs = []
                        else:
                            raw, cell_repairs = _repair_numeric_cell_vertical_splits(table, raw)
                        rows = [[("" if c is None else str(c)).strip() for c in r] for r in raw]
                        rows = [r for r in rows if any(r)]
                        if not rows:
                            continue
                        parsed_table = ParsedTable(
                            page=page_no, headers=rows[0], rows=rows[1:]
                        )
                        tables.append(parsed_table)
                        table_elements.append(
                            DocumentElement(
                                page_no=page_no,
                                element_type="table",
                                text=parsed_table.to_text(),
                                bbox=tuple(float(value) for value in table.bbox),
                                table_cells=rows,
                                parser="pdfplumber",
                                source_ref=f"page:{page_no}:table:{len(tables)}",
                                metadata={"cell_repairs": cell_repairs} if cell_repairs else {},
                            )
                        )
                except Exception as exc:  # pdfplumber 解析失败不影响文本
                    logger.warning(
                        "table extract failed page=%s type=%s",
                        page_no,
                        type(exc).__name__,
                    )
                text = _page_text_without_tables(fz_page, table_bboxes).strip()
                scanned = len(text.replace(" ", "")) < SCAN_PAGE_CHARS
                elements = []
                if text:
                    elements.append(
                        DocumentElement(
                            page_no=page_no,
                            element_type="text",
                            text=text,
                            bbox=tuple(float(value) for value in fz_page.rect),
                            reading_order=0,
                            parser="pymupdf",
                            source_ref=f"page:{page_no}:text",
                            metadata={"granularity": "page"},
                        )
                    )
                for index, element in enumerate(table_elements, start=len(elements)):
                    element.reading_order = index
                pages.append(
                    ParsedPage(
                        page_no=page_no,
                        text=text,
                        tables=tables,
                        scanned=scanned,
                        elements=elements + table_elements,
                    )
                )
    rate = _extraction_rate(pages)
    logger.info(
        "pdf parsed: pages=%d tables=%d rate=%.2f",
        len(pages),
        sum(len(p.tables) for p in pages),
        rate,
    )
    return LayoutResult(pages=pages, text_extraction_rate=rate)


def _is_good_table(table) -> bool:
    """表格是否"有效"（可作为结构化财报表）。

    仅是 :func:`_is_good_rows` 的薄包装，保持 lines 策略（stage1）调用点不变。
    text-rescue 路径在 P3.1 重建 rows 后改调 :func:`_is_good_rows`。
    """
    try:
        return _is_good_rows(table.extract() or [])
    except Exception:
        return False


def _is_good_rows(rows_in) -> bool:
    """rows 级判定，与 :func:`_is_good_table` 同规则，供 word-atomic 重建后调用。


    多信号判定，不再把"表头含年份"当作唯一硬条件——否则无框线年度财报表
    （如 WFC Table 9）因表头是叙述（无独立年份 header）而被整体丢弃，导致
    数值列整块不进语料。

    **P2 回归（Dev V2 ablation, FIRST_REGRESSION_COMMIT=bcbf189）修复：**
    P2 把判定收紧成"裸年份 fullmatch 或 纯数值密度"，却拒绝了年份**嵌在表头
    短语里**的财报表（如"截至2025年"、多级表头分部表），使利润表/无框线分部表
    整页退化成 TEXT，数值行被 chunk 噪声过滤丢弃。本函数放宽 S2，仍保持
    S1+S3 防回归（WFC/海尔/神华）。

    - S1 表头首行含**独立年份 token**（如 ``2025``）→ 强加分，直接有效（兼容有线表）；
    - S3 由**纯数值 cell 密度**决定：每行 ≥4 个"纯数值 cell"的行占比 ≥0.25，且列数 ≥4
      （保留，杜绝在 Dev 集上移除回归；其排除双栏正文页的意图由 ncols<3 继续兜底）；
    - S2 新增：**表头区域（top ≤6 行）含年份 token**（允许"2025年"／"截至2025年"，
      非裸年份也可）且 **列数 ≥3** 且 **≥2 个数据行各含 ≥2 个数值 cell** → 有效
      （修复 Dev V2 A+B：嵌入年份的利润表、无框线/多级表头分部表、宽表）。

    用"纯数值 cell"而非"任一含数字的 cell"，是因为普通双栏正文页的文本
    片段也含零星数字（如 ``as of December 31, 2025``），但不会形成一行 2+ 个
    独立对齐的数值 token（真财务表的数据行才是那样）。叙述/双栏正文页在 text
    策略下多为 1-2 列垃圾表，被 ncols/数值行门槛拒绝。
    """
    try:
        rows = list(rows_in or [])
    except Exception:
        return False
    import re

    rows = [r for r in rows if any(r)]
    if not rows or len(rows[0]) < 3:
        return False
    ncols = len(rows[0])
    n = len(rows)

    # S1: 表头首行含独立年份 token（完整年份，非叙述里的年份）
    if any(re.fullmatch(r"(19|20)\d{2}", str(h).strip()) for h in rows[0]):
        return True

    # S3: 纯数值 cell 密度（现行，保留以杜绝 Dev 集移除回归）
    pure_num_rows = 0
    for row in rows:
        pure = 0
        for cell in row:
            text = str(cell).strip()
            if text and re.search(r"\d", text) and not re.search(r"[a-zA-Z]", text):
                pure += 1
        if pure >= 4:
            pure_num_rows += 1
    dense = pure_num_rows / n if n else 0.0
    if ncols >= 4 and dense >= 0.25:
        return True

    # S2: 表头区域含年份 token（允许"2025年"/"截至2025年"，非裸年份）+ 数值数据行
    header_rows = rows[: min(6, n)]
    year_in_header = False
    for row in header_rows:
        for cell in row:
            text = str(cell).strip()
            if text and re.search(r"(19|20)\d{2}", text) and len(text) <= 30 \
               and not re.search(r"[a-zA-Z]{5,}", text):
                year_in_header = True
                break
        if year_in_header:
            break
    numeric_body_rows = 0
    for row in rows:
        ncells = 0
        for cell in row:
            text = str(cell).strip()
            if text and re.search(r"\d", text) and not re.search(r"[a-zA-Z]", text):
                ncells += 1
        if ncells >= 2:
            numeric_body_rows += 1
    if year_in_header and ncols >= 3 and numeric_body_rows >= 2:
        return True

    return False


class _WordAtomicTable:
    """P3.1：把 text-rescue 候选的 cells 换成 word-atomic 重建结果后的轻量代理。

    只代理 ``parse_pdf`` / ``_repair_numeric_cell_vertical_splits`` 实际用到的接口
    （``bbox`` / ``extract`` / ``rows`` / ``page``），其余一律回落原对象。
    ``word_atomic=True`` 供调用方跳过已冗余的 P1 数值切分修复。
    """

    word_atomic = True

    def __init__(self, origin, rows: list[list[str]]) -> None:
        self._origin = origin
        self._rows = rows

    @property
    def bbox(self):
        return self._origin.bbox

    @property
    def rows(self):
        return self._origin.rows

    @property
    def columns(self):
        return self._origin.columns

    @property
    def cells(self):
        return self._origin.cells

    @property
    def page(self):
        return self._origin.page

    def extract(self):
        return [list(r) for r in self._rows]


def _rebuild_rows_word_atomic(table, words: list[dict]) -> list[list[str]] | None:
    """P3.1 ``WORD_ATOMIC_RESCUE_CELL_RECONSTRUCTION``：用完整 word 重建 cells。

    命名
    ----
    canonical name  = ``WORD_ATOMIC_RESCUE_CELL_RECONSTRUCTION``
    superseded name = ``WORD_ATOMIC_COLUMN_BOUNDARY_REPAIR``
    supersede reason = boundary-level invariant（"任何推断列边界不得穿过 word bbox"）
        在几何上不可行，实现已改为 word-atomic cell reconstruction。详见
        ``dev_v3_p31_naming_supersession_v1.json``。

    scope（不变量适用范围）
    ----------------------
    **仅限 stage2 text/text rescue 路径。**
    被分配到重建栅格的 source word 原子地归属单个 cell，不被字符切分。
    stage1（lines strategy）不经过本函数，其残留的 word-split 登记为
    ``OUT_OF_SCOPE_KNOWN_RESIDUAL``，不计为 P3.1 failure。

    不变式
    ------
        A word is never split across cells; it is assigned wholly to one column.

    为什么不是"移动列边界"
    ----------------------
    实测 Morgan Stanley p187：text/text rescue 候选的 8 条内部列边界 **100%** 穿过
    source word bbox；而左侧叙述文本跨行连通成一片，导致"全局竖线不切任何 word"
    在数学上不可满足——把边界 snap 到 word 外缘会一路滑到表格边界（实测
    ``113.895 -> 38.25``），摧毁列结构。故 P3.1 **不改列边界**，而是改 cell 的
    构造方式：cell 文本由完整 word 归属而成，而不是让竖线按字符劈开 word。

    典型修复
    --------
        before: ['202', '5']          （word "2025" 被 x=414.844 切开）
        after : ['2025', '']
        before: ['Net income applicabl', 'e to noncont', 'rolling interests']
        after : ['Net income applicable', 'to noncontrolling', 'interests']

    归属规则：word 中心点落入哪个 (行 band, 列 band) 就整词归到该 cell。
    任一步拿不到几何证据（cells 为空 / 无 words / 网格退化）都返回 ``None``
    由调用方回退到原 rows，绝不静默造数。
    """
    try:
        cells = [c for c in (table.cells or []) if c is not None and len(c) >= 4]
    except Exception:
        return None
    if not cells or not words:
        return None

    xs = sorted({round(float(c[0]), 3) for c in cells} | {round(float(c[2]), 3) for c in cells})
    ys = sorted({round(float(c[1]), 3) for c in cells} | {round(float(c[3]), 3) for c in cells})
    if len(xs) < 3 or len(ys) < 2:
        return None

    grid: list[list[list[str]]] = [[[] for _ in range(len(xs) - 1)] for _ in range(len(ys) - 1)]
    assigned = 0
    for w in words:
        try:
            wx0, wx1 = float(w["x0"]), float(w["x1"])
            wtop = float(w.get("top", w.get("doctop", 0.0)))
            wbot = float(w.get("bottom", w.get("y1", wtop)))
        except (KeyError, TypeError, ValueError):
            continue
        wxc = (wx0 + wx1) / 2.0
        wyc = (wtop + wbot) / 2.0

        ri = next(
            (i for i in range(len(ys) - 1) if ys[i] - 0.5 <= wyc <= ys[i + 1] + 0.5),
            None,
        )
        if ri is None:
            continue
        ci = next(
            (j for j in range(len(xs) - 1) if xs[j] - 0.5 <= wxc <= xs[j + 1] + 0.5),
            None,
        )
        if ci is None:
            continue
        grid[ri][ci].append(str(w.get("text") or "").strip())
        assigned += 1

    if not assigned:
        return None
    rows = [[" ".join(t for t in cell if t).strip() for cell in row] for row in grid]
    rows = [r for r in rows if any(r)]
    if not rows:
        return None
    return rows


def _repair_numeric_cell_vertical_splits(table, raw: list) -> tuple[list, list]:
    """修复 pdfplumber text 策略对无框线财务表的 numeric-cell vertical split。

    背景：无框线财报主表走 ``vertical_strategy="text"`` 时，pdfplumber 用文本 x
    对齐估计列边界，常把带逗号数值的中间误设为列分隔线，使一个完整数值 word
    被纵切成两个相邻 cell（如 '83,6' + '99'，实为 '83,699'）。

    本函数是 **geometry / schema-aware 定向修复**，而非全局字符串启发式：
    只有当相邻两 cell 的分界 ``x_split`` 严格落在某个原始数值 word 的 x-bbox
    内部（``word.x0 < x_split < word.x1``），且左右片段的字符模式与合并值都符合
    合法数字，才合并，并且重建值直接采用该原始 word 的文本（保留年份无逗号、
    财务数字带逗号的真实格式）。任一无几何证据的"看起来能拼"都不生效。

    返回 ``(repaired_raw, audits)``；``audits`` 逐条记录
    ``repair_type / original_cells / reconstructed_value / x_split / word_bbox``
    以及 P1.1 guard 证据 ``word_yband / row_yband / straddle_matches / n_candidates``，
    以便审计，绝不静默造数。

    **P1.1 两大 guard（防 wrong-row / ambiguous）：**
    - ``Y-overlap guard``：候选 word 与该行 band 须有实质纵向重叠（>=word 高 30%），
      否则该 word 来自另一行同名数值（wrong-row），一律拒绝。
    - ``uniqueness guard``：过滤后候选必须恰好 1 个；0 或 >1 视为 ambiguous，放弃修复。
    已在 WFC 全文档验证：112 条修复的数值集合与无 guard 时完全一致（纯安全网，
    不改数值），仅收紧 provenance 并拒掉 4 条同值错行候选。
    """
    try:
        words = [w for w in table.page.extract_words() if w.get("text")]
    except Exception:
        return raw, []
    if not words:
        return raw, []
    import re

    def fin_partial(value: str) -> bool:
        v = value.strip().lstrip("$").strip()
        return bool(re.fullmatch(r"\d{1,3}(,\d{1,3})*", v))

    def fin_digits(value: str) -> bool:
        return bool(re.fullmatch(r"\d+", value.strip()))

    out = [list(r) for r in raw]
    audits: list[dict] = []

    def word_top(w) -> float | None:
        return w.get("top", w.get("y0", w.get("doctop")))

    def word_bottom(w) -> float | None:
        return w.get("bottom", w.get("y1", w.get("doctop")))

    for i in range(len(out)):
        row = out[i]
        if i >= len(table.rows):
            break
        cells = table.rows[i].cells
        if cells is None or len(cells) != len(row):
            continue
        # 行的 y-band（由该行所有 cell bbox 计算，跨整行高度）
        cell_y = [c for c in cells if c is not None and len(c) >= 4]
        if not cell_y:
            continue
        row_y0 = min(float(c[1]) for c in cell_y)
        row_y1 = max(float(c[3]) for c in cell_y)
        j = 0
        while j < len(row) - 1:
            left = str(row[j]).strip()
            right = str(row[j + 1]).strip()
            if fin_partial(left) and fin_digits(right):
                x_split = float(cells[j][2])  # 左 cell 的右边界 = 切分位置
                merged = (left.lstrip("$") + right).replace(",", "")
                if re.fullmatch(r"\d+", merged):
                    straddle_count = 0
                    candidates: list[dict] = []
                    for w in words:
                        wtext = w.get("text") or ""
                        if (
                            wtext.replace(",", "") == merged
                            and float(w["x0"]) < x_split - 0.5
                            and float(w["x1"]) > x_split + 0.5
                        ):
                            straddle_count += 1
                            wtop = word_top(w)
                            wbot = word_bottom(w)
                            if wtop is None or wbot is None or wbot <= wtop:
                                continue
                            # Y-overlap guard：与行 band 须有实质纵向重叠，
                            # 否则该 word 来自另一行（wrong-row）。
                            shared_lo = max(wtop, row_y0)
                            shared_hi = min(wbot, row_y1)
                            overlap = shared_hi - shared_lo
                            if overlap <= 0 or overlap < 0.3 * (wbot - wtop):
                                continue
                            candidates.append(w)
                    # uniqueness guard：恰好一个候选才允许修复（0 或 >1 视为 ambiguous）
                    if len(candidates) == 1:
                        w = candidates[0]
                        row[j] = w.get("text")
                        row[j + 1] = ""
                        audits.append(
                            {
                                "repair_type": "numeric_cell_vertical_split",
                                "row_index": i,
                                "cell_index": j,
                                "original_cells": [left, right],
                                "reconstructed_value": w.get("text"),
                                "x_split": round(x_split, 2),
                                "word_bbox": [round(float(w["x0"]), 2), round(float(w["x1"]), 2)],
                                "word_yband": [round(word_top(w), 2), round(word_bottom(w), 2)],
                                "row_yband": [round(row_y0, 2), round(row_y1, 2)],
                                "straddle_matches": straddle_count,
                                "wrong_row_rejected": straddle_count - 1,
                                "n_candidates": 1,
                            }
                        )
            j += 1
    return out, audits


def _bbox_overlap(a, b) -> bool:
    """两个 bbox 是否重叠（用于去重 text/line 策略检测到的同一区域）。"""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua >= 0.5 if ua else False


def _page_text_without_tables(
    fz_page, table_bboxes: list[tuple[float, float, float, float]]
) -> str:
    """页面文本，剔除已识别表格区域内的词（避免表格内容双重入库）。失败回退全量文本。

    无表格的页面先做双栏检测：双栏正文若按文档流顺序提取会左右栏交错，
    需按"先左栏后右栏（栏内按 y）"重组；非双栏回退默认文本流。
    """
    if not table_bboxes:
        two_col = _two_column_text(fz_page)
        if two_col is not None:
            return two_col
        return fz_page.get_text("text")
    try:
        words = fz_page.get_text("words")  # x0,y0,x1,y1,word,block,line,word_no
        if not words:
            return fz_page.get_text("text")
        keep = [
            w
            for w in words
            if not any(
                bx0 <= w[0] and w[2] <= bx1 and by0 <= w[1] and w[3] <= by1
                for (bx0, by0, bx1, by1) in table_bboxes
            )
        ]
        if not keep:
            return ""
        # 按视觉行（block, line）重组，保持原阅读顺序
        lines: dict[tuple[int, int], list[str]] = {}
        for w in sorted(keep, key=lambda w: (w[5], w[6], w[0])):
            lines.setdefault((w[5], w[6]), []).append(w[4])
        return "\n".join(" ".join(v) for v in lines.values())
    except Exception as exc:  # 掩码失败不影响解析
        logger.warning("table mask failed, fallback full text: type=%s", type(exc).__name__)
        return fz_page.get_text("text")


def _two_column_text(fz_page) -> str | None:
    """启发式双栏检测并重组文本；非双栏返回 None（调用方回退默认流程）。

    使用 line 级 bbox（fitz get_text("dict") 的 lines），避免 fitz 将同行左右文本
    合并进同一 block 导致的误判。判定条件（全部满足才视为双栏）：
    - 窄行（宽度 < 页宽 60%）>= 6 个；
    - 以页面中线为界，左右两侧各至少 20% 的窄行；
    - 左右栏平均中心间距 >= 页宽 25%（确认存在明显分栏缝）。
    """
    try:
        page_w = fz_page.rect.width or 1.0
        lines: list[tuple[tuple[float, float, float, float], str]] = []
        for blk in fz_page.get_text("dict").get("blocks", []):
            if blk.get("type") != 0:
                continue
            for ln in blk.get("lines", []):
                bbox = ln.get("bbox")
                txt = " ".join((s.get("text") or "") for s in ln.get("spans", [])).strip()
                if txt and bbox and (bbox[2] - bbox[0]) < page_w * 0.6:
                    lines.append((bbox, txt))
        if len(lines) < 6:
            return None
        mid = page_w / 2
        left = [ln for ln in lines if (ln[0][0] + ln[0][2]) / 2 < mid]
        right = [ln for ln in lines if (ln[0][0] + ln[0][2]) / 2 >= mid]
        if not left or not right:
            return None
        if len(left) / len(lines) < 0.2 or len(right) / len(lines) < 0.2:
            return None
        ml = sum((ln[0][0] + ln[0][2]) / 2 for ln in left) / len(left)
        mr = sum((ln[0][0] + ln[0][2]) / 2 for ln in right) / len(right)
        if mr - ml < page_w * 0.25:
            return None
        # 先左栏（栏内按 y 自上而下）、再右栏
        ordered = sorted(left, key=lambda ln: ln[0][1]) + sorted(right, key=lambda ln: ln[0][1])
        return "\n".join(t for _, t in ordered)
    except Exception as exc:  # 检测失败不影响解析
        logger.warning("two-column detect failed, fallback: type=%s", type(exc).__name__)
        return None


def parse_docx(path: Path) -> LayoutResult:
    import docx

    d = docx.Document(path)
    text_parts: list[str] = []
    tables: list[ParsedTable] = []
    elements: list[DocumentElement] = []
    for para in d.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)
            style_name = str(getattr(para.style, "name", "") or "")
            heading_level = None
            if style_name.casefold().startswith("heading "):
                suffix = style_name.rsplit(" ", 1)[-1]
                heading_level = int(suffix) if suffix.isdigit() else None
            elements.append(
                DocumentElement(
                    page_no=1,
                    element_type="heading" if heading_level is not None else "text",
                    text=para.text,
                    reading_order=len(elements),
                    heading_level=heading_level,
                    parser="python-docx",
                    source_ref=f"paragraph:{len(text_parts)}",
                )
            )
    for ti, t in enumerate(d.tables):
        rows = [[c.text.strip() for c in row.cells] for row in t.rows]
        rows = [r for r in rows if any(r)]
        if rows:
            parsed_table = ParsedTable(page=1, headers=rows[0], rows=rows[1:])
            tables.append(parsed_table)
            elements.append(
                DocumentElement(
                    page_no=1,
                    element_type="table",
                    text=parsed_table.to_text(),
                    reading_order=len(elements),
                    table_cells=rows,
                    parser="python-docx",
                    source_ref=f"table:{ti + 1}",
                )
            )
    pages = [
        ParsedPage(
            page_no=1,
            text="\n".join(text_parts),
            tables=tables,
            elements=elements,
        )
    ]
    return LayoutResult(pages=pages, text_extraction_rate=1.0)


def parse_xlsx(path: Path) -> LayoutResult:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    pages: list[ParsedPage] = []
    for ws in wb.worksheets:
        page_no = len(pages) + 1
        rows: list[list[str]] = []
        for row in ws.iter_rows(values_only=True):
            vals = ["" if v is None else str(v) for v in row]
            if any(vals):
                rows.append(vals)
        tables: list[ParsedTable] = []
        elements = [
            DocumentElement(
                page_no=page_no,
                element_type="heading",
                text=f"工作表: {ws.title}",
                reading_order=0,
                heading_level=1,
                parser="openpyxl",
                source_ref=f"sheet:{ws.title}",
                metadata={"sheet": ws.title},
            )
        ]
        if rows:
            parsed_table = ParsedTable(
                page=page_no, headers=rows[0], rows=rows[1:]
            )
            tables.append(parsed_table)
            elements.append(
                DocumentElement(
                    page_no=page_no,
                    element_type="table",
                    text=parsed_table.to_text(),
                    reading_order=1,
                    table_cells=rows,
                    parser="openpyxl",
                    source_ref=f"sheet:{ws.title}",
                    metadata={"sheet": ws.title},
                )
            )
        pages.append(
            ParsedPage(
                page_no=page_no,
                text=f"工作表: {ws.title}",
                tables=tables,
                elements=elements,
            )
        )
    return LayoutResult(pages=pages, text_extraction_rate=1.0)


def parse_image(path: Path) -> LayoutResult:
    """图片（截图/扫描件）经 OCR 转文本（需 OCR_ENABLED=true）。"""
    from app.parsers.ocr import ocr_image_file

    text = ocr_image_file(path)
    elements = [
        DocumentElement(
            page_no=1,
            element_type="text",
            text=text,
            reading_order=0,
            parser="paddleocr",
            source_ref="page:1:ocr",
        )
    ] if text else []
    page = ParsedPage(page_no=1, text=text, elements=elements)
    scanned = not bool(text.strip())
    return LayoutResult(pages=[page], text_extraction_rate=0.0 if scanned else 1.0)


def _extraction_rate(pages: list[ParsedPage]) -> float:
    if not pages:
        return 1.0
    with_text = sum(1 for p in pages if not p.scanned and len(p.text.replace(" ", "")) >= SCAN_PAGE_CHARS)
    return with_text / len(pages)
