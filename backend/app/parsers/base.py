"""解析层抽象：版式层 / OCR 层 / 结构层 / 元数据层 的分层调度入口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import get_settings

CURRENT_LAYOUT_SCHEMA_VERSION = 2
from app.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg"}


@dataclass
class DocumentElement:
    """Parser-neutral document element with traceable page provenance.

    Unknown layout attributes remain ``None`` rather than receiving invented
    coordinates or confidence scores.  This lets fast parsers and richer
    fallback parsers share one loss-aware interchange format.
    """

    page_no: int
    element_type: str
    text: str = ""
    bbox: tuple[float, float, float, float] | None = None
    reading_order: int = 0
    heading_level: int | None = None
    table_cells: list[list[str]] = field(default_factory=list)
    parser: str = "unknown"
    confidence: float | None = None
    source_ref: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "page_no": self.page_no,
            "element_type": self.element_type,
            "text": self.text,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "reading_order": self.reading_order,
            "heading_level": self.heading_level,
            "table_cells": self.table_cells,
            "parser": self.parser,
            "confidence": self.confidence,
            "source_ref": self.source_ref,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentElement":
        bbox = data.get("bbox")
        return cls(
            page_no=int(data["page_no"]),
            element_type=str(data.get("element_type") or "text"),
            text=str(data.get("text") or ""),
            bbox=tuple(float(value) for value in bbox) if bbox is not None else None,
            reading_order=int(data.get("reading_order") or 0),
            heading_level=(
                int(data["heading_level"])
                if data.get("heading_level") is not None
                else None
            ),
            table_cells=[
                ["" if cell is None else str(cell) for cell in row]
                for row in (data.get("table_cells") or [])
            ],
            parser=str(data.get("parser") or "unknown"),
            confidence=(
                float(data["confidence"])
                if data.get("confidence") is not None
                else None
            ),
            source_ref=str(data.get("source_ref") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ParsedTable:
    """表格单元（作为不可分割块）。"""

    page: int
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)

    def to_text(self) -> str:
        lines: list[str] = []
        if self.headers:
            lines.append(" | ".join(str(h) for h in self.headers))
        for r in self.rows:
            lines.append(" | ".join("" if c is None else str(c) for c in r))
        return "\n".join(lines)


@dataclass
class ParsedPage:
    page_no: int
    text: str = ""
    tables: list[ParsedTable] = field(default_factory=list)
    scanned: bool = False  # 文本过少，标记为需 OCR
    elements: list[DocumentElement] = field(default_factory=list)

    def normalized_elements(self) -> list[DocumentElement]:
        """Return explicit elements, or a faithful legacy projection."""

        if self.elements:
            return self.elements
        output: list[DocumentElement] = []
        if self.text:
            output.append(
                DocumentElement(
                    page_no=self.page_no,
                    element_type="text",
                    text=self.text,
                    reading_order=0,
                    parser="legacy-layout",
                    source_ref=f"page:{self.page_no}",
                )
            )
        for index, table in enumerate(self.tables, start=len(output)):
            cells = ([table.headers] if table.headers else []) + table.rows
            output.append(
                DocumentElement(
                    page_no=self.page_no,
                    element_type="table",
                    text=table.to_text(),
                    reading_order=index,
                    table_cells=cells,
                    parser="legacy-layout",
                    source_ref=f"page:{self.page_no}:table:{index + 1}",
                )
            )
        return output


@dataclass
class LayoutResult:
    """版式层输出。"""

    pages: list[ParsedPage]
    text_extraction_rate: float = 1.0  # 含文本页数 / 总页数

    @property
    def document_elements(self) -> list[DocumentElement]:
        return [
            element
            for page in self.pages
            for element in page.normalized_elements()
        ]

    def to_dict(self) -> dict:
        return {
            "schema_version": CURRENT_LAYOUT_SCHEMA_VERSION,
            "pages": [
                {
                    "page_no": p.page_no,
                    "text": p.text,
                    "tables": [
                        {"page": t.page, "headers": t.headers, "rows": t.rows}
                        for t in p.tables
                    ],
                    "scanned": p.scanned,
                    "elements": [element.to_dict() for element in p.normalized_elements()],
                }
                for p in self.pages
            ],
            "text_extraction_rate": self.text_extraction_rate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LayoutResult":
        pages = [
            ParsedPage(
                page_no=p["page_no"],
                text=p["text"],
                tables=[
                    ParsedTable(page=t["page"], headers=t["headers"], rows=t["rows"])
                    for t in p["tables"]
                ],
                scanned=p["scanned"],
                elements=[
                    DocumentElement.from_dict(element)
                    for element in (p.get("elements") or [])
                ],
            )
            for p in d["pages"]
        ]
        return cls(pages=pages, text_extraction_rate=d["text_extraction_rate"])


def layout_payload_is_current(payload: dict | None) -> bool:
    """Return whether a cached layout contains the loss-aware element schema."""

    if not isinstance(payload, dict):
        return False
    if payload.get("schema_version") != CURRENT_LAYOUT_SCHEMA_VERSION:
        return False
    pages = payload.get("pages")
    return isinstance(pages, list) and all(
        isinstance(page, dict) and isinstance(page.get("elements"), list)
        for page in pages
    )


def parse_layout(path: Path) -> LayoutResult:
    """按扩展名分发到具体解析器。"""
    ext = path.suffix.lower()
    if ext == ".pdf":
        from app.parsers.layout import parse_pdf

        return parse_pdf(path)
    if ext in (".docx",):
        from app.parsers.layout import parse_docx

        return parse_docx(path)
    if ext in (".xlsx",):
        from app.parsers.layout import parse_xlsx

        return parse_xlsx(path)
    if ext in (".png", ".jpg", ".jpeg"):
        if not get_settings().ocr_enabled:
            raise ValueError("图片类型需 OCR 解析，请先启用 OCR（OCR_ENABLED=true 并配置 OCR 密钥）")
        from app.parsers.layout import parse_image

        return parse_image(path)
    raise ValueError(f"不支持的文件类型: {ext}（支持 {sorted(SUPPORTED_EXTENSIONS)}）")
