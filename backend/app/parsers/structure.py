"""结构层：从版式层结果构建章节树，并标注每页所属章节。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.parsers.base import LayoutResult

# (pattern, level)；level 为 None 表示按点号数计算（1.2.3 → 3）
_HEADING_PATTERNS: list[tuple[re.Pattern, int | None]] = [
    (re.compile(r"^第[一二三四五六七八九十百]+[章节部分篇]\s*(.{1,40})$"), 1),
    (re.compile(r"^[一二三四五六七八九十百]+、\s*(.{1,40})$"), 1),  # 一、营业收入
    (re.compile(r"^(\d{1,2}(\.\d{1,2}){0,3})\s+(.{1,60})$"), None),  # 1.2.3 标题
    (re.compile(r"^[（(][一二三四五六七八九十0-9]+[)）]\s*(.{1,60})$"), 2),
]


@dataclass
class Section:
    title: str
    level: int
    page: int
    path: str = ""  # 章节路径：主标题 > 子标题
    seq: int = 0

    def to_dict(self) -> dict:
        return {"title": self.title, "level": self.level, "page": self.page, "path": self.path, "seq": self.seq}

    @classmethod
    def from_dict(cls, d: dict) -> "Section":
        return cls(title=d["title"], level=d["level"], page=d["page"], path=d.get("path", ""), seq=d.get("seq", 0))


def _looks_like_heading(line: str) -> int | None:
    """返回标题层级（1~3），非标题返回 None。"""
    line = line.strip()
    if not line or len(line) > 80:
        return None
    if line.isupper() and len(line) <= 40:  # 全大写短行
        return 1
    for pat, level in _HEADING_PATTERNS:
        m = pat.match(line)
        if m:
            if level is not None:
                return level
            num = m.group(1)
            return num.count(".") + 1
    return None


def build_section_tree(layout: LayoutResult) -> list[Section]:
    """按页扫描文本，用标题模式推断章节，输出扁平章节列表（含层级）。"""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    seq = 0
    for page in layout.pages:
        for raw_line in page.text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            level = _looks_like_heading(line)
            if level is not None:
                title = line
                # 维持层级栈：同层覆盖，深层入栈
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                seq += 1
                path = " > ".join(t for _, t in stack)
                sections.append(
                    Section(title=title, level=level, page=page.page_no, path=path, seq=seq)
                )
    if not sections:
        # 无标题时以"文档正文"兜底
        sections.append(Section(title="文档正文", level=1, page=1, path="文档正文", seq=1))
    return sections


def section_path_for_page(sections: list[Section], page_no: int) -> str:
    """取该页最后（最深的）章节路径。"""
    path = ""
    for s in sections:
        if s.page <= page_no:
            path = s.path
        else:
            break
    return path or (sections[0].path if sections else "")
