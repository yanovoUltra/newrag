"""解析层抽象：版式层 / OCR 层 / 结构层 / 元数据层 的分层调度入口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg"}


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


@dataclass
class LayoutResult:
    """版式层输出。"""

    pages: list[ParsedPage]
    text_extraction_rate: float = 1.0  # 含文本页数 / 总页数

    def to_dict(self) -> dict:
        return {
            "pages": [
                {
                    "page_no": p.page_no,
                    "text": p.text,
                    "tables": [
                        {"page": t.page, "headers": t.headers, "rows": t.rows}
                        for t in p.tables
                    ],
                    "scanned": p.scanned,
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
            )
            for p in d["pages"]
        ]
        return cls(pages=pages, text_extraction_rate=d["text_extraction_rate"])


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
