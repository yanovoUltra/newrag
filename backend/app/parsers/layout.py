"""版式层：PyMuPDF / PDFPlumber 提取文本与表格。"""

from __future__ import annotations

import io
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
                text = fz_page.get_text("text").strip()
                tables: list[ParsedTable] = []
                try:
                    for raw in pl_page.extract_tables() or []:
                        if not raw:
                            continue
                        rows = [[("" if c is None else str(c)).strip() for c in r] for r in raw]
                        rows = [r for r in rows if any(r)]
                        if not rows:
                            continue
                        tables.append(
                            ParsedTable(page=page_no, headers=rows[0], rows=rows[1:])
                        )
                except Exception as e:  # pdfplumber 解析失败不影响文本
                    logger.warning("table extract failed page %s: %s", page_no, e)
                scanned = len(text.replace(" ", "")) < SCAN_PAGE_CHARS
                pages.append(
                    ParsedPage(page_no=page_no, text=text, tables=tables, scanned=scanned)
                )
    rate = _extraction_rate(pages)
    logger.info("pdf parsed: %s pages=%d tables=%d rate=%.2f", path.name, len(pages), sum(len(p.tables) for p in pages), rate)
    return LayoutResult(pages=pages, text_extraction_rate=rate)


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


def _extraction_rate(pages: list[ParsedPage]) -> float:
    if not pages:
        return 1.0
    with_text = sum(1 for p in pages if not p.scanned and len(p.text.replace(" ", "")) >= SCAN_PAGE_CHARS)
    return with_text / len(pages)


def export_pdf_text(path: Path) -> str:
    """辅助：纯文本导出。"""
    return parse_pdf(path).all_text()
