"""版式层：PyMuPDF / PDFPlumber 提取文本与表格。"""

from __future__ import annotations

from pathlib import Path

from app.core.logging import get_logger
from app.parsers.base import LayoutResult, ParsedPage, ParsedTable

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
                        for t in text_tables:
                            # 只保留"有效"财报表（≥3列且表头含年份）；text 策略会把普通正文/整列数值
                            # 挤成单格垃圾表，若加入会屏蔽正文文本并污染结构化表，故丢弃。
                            if not _is_good_table(t):
                                continue
                            if any(_bbox_overlap(t.bbox, b) for b in table_bboxes):
                                continue
                            found.append(t)
                            table_bboxes.append(t.bbox)
                    for table in found:
                        raw = table.extract() or []
                        rows = [[("" if c is None else str(c)).strip() for c in r] for r in raw]
                        rows = [r for r in rows if any(r)]
                        if not rows:
                            continue
                        tables.append(
                            ParsedTable(page=page_no, headers=rows[0], rows=rows[1:])
                        )
                except Exception as e:  # pdfplumber 解析失败不影响文本
                    logger.warning("table extract failed page %s: %s", page_no, e)
                text = _page_text_without_tables(fz_page, table_bboxes).strip()
                scanned = len(text.replace(" ", "")) < SCAN_PAGE_CHARS
                pages.append(
                    ParsedPage(page_no=page_no, text=text, tables=tables, scanned=scanned)
                )
    rate = _extraction_rate(pages)
    logger.info("pdf parsed: %s pages=%d tables=%d rate=%.2f", path.name, len(pages), sum(len(p.tables) for p in pages), rate)
    return LayoutResult(pages=pages, text_extraction_rate=rate)


def _is_good_table(table) -> bool:
    """表格是否"有效"：≥3 列且表头含年份（可作为结构化财报表）。"""
    try:
        rows = table.extract() or []
    except Exception:
        return False
    rows = [r for r in rows if any(r)]
    if not rows or len(rows[0]) < 3:
        return False
    import re

    return any(re.search(r"(19|20)\d{2}", str(h) or "") for h in rows[0])


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
    except Exception as e:  # 掩码失败不影响解析
        logger.warning("table mask failed, fallback full text: %s", e)
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
    except Exception as e:  # 检测失败不影响解析
        logger.warning("two-column detect failed, fallback: %s", e)
        return None


def parse_docx(path: Path) -> LayoutResult:
    import docx

    d = docx.Document(path)
    text_parts: list[str] = []
    tables: list[ParsedTable] = []
    for para in d.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)
    for ti, t in enumerate(d.tables):
        rows = [[c.text.strip() for c in row.cells] for row in t.rows]
        rows = [r for r in rows if any(r)]
        if rows:
            tables.append(ParsedTable(page=1, headers=rows[0], rows=rows[1:]))
    pages = [ParsedPage(page_no=1, text="\n".join(text_parts), tables=tables)]
    return LayoutResult(pages=pages, text_extraction_rate=1.0)


def parse_xlsx(path: Path) -> LayoutResult:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    pages: list[ParsedPage] = []
    tables: list[ParsedTable] = []
    for ws in wb.worksheets:
        rows: list[list[str]] = []
        for row in ws.iter_rows(values_only=True):
            vals = ["" if v is None else str(v) for v in row]
            if any(vals):
                rows.append(vals)
        if rows:
            tables.append(ParsedTable(page=len(pages) + 1, headers=rows[0], rows=rows[1:]))
        pages.append(ParsedPage(page_no=len(pages) + 1, text=f"工作表: {ws.title}", tables=[]))
    # 表格作为唯一内容
    if tables:
        pages = [ParsedPage(page_no=1, text="", tables=tables)]
    return LayoutResult(pages=pages, text_extraction_rate=1.0)


def parse_image(path: Path) -> LayoutResult:
    """图片（截图/扫描件）经 OCR 转文本（需 OCR_ENABLED=true）。"""
    from app.parsers.ocr import ocr_image_file

    text = ocr_image_file(path)
    page = ParsedPage(page_no=1, text=text)
    scanned = not bool(text.strip())
    return LayoutResult(pages=[page], text_extraction_rate=0.0 if scanned else 1.0)


def _extraction_rate(pages: list[ParsedPage]) -> float:
    if not pages:
        return 1.0
    with_text = sum(1 for p in pages if not p.scanned and len(p.text.replace(" ", "")) >= SCAN_PAGE_CHARS)
    return with_text / len(pages)
