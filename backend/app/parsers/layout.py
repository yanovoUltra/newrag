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


def _page_text_without_tables(
    fz_page, table_bboxes: list[tuple[float, float, float, float]]
) -> str:
    """页面文本，剔除已识别表格区域内的词（避免表格内容双重入库）。失败回退全量文本。"""
    if not table_bboxes:
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
