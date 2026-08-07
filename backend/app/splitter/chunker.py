"""Small-to-Big 切分：叶子 chunk 128~256 token（相邻重叠 10%），表格不可分割。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.parsers.base import LayoutResult
from app.parsers.structure import Section, section_path_for_page
from app.splitter.tokens import count_tokens, split_text_by_tokens


@dataclass
class Chunk:
    id: str
    doc_id: str
    page: int
    section_path: str
    chunk_type: str  # text | table | chart
    content: str
    token_count: int
    seq: int = 0
    parent_id: str | None = None  # 父节点（章节级），阶段二用于上下文扩展

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "page": self.page,
            "section_path": self.section_path,
            "chunk_type": self.chunk_type,
            "content": self.content,
            "token_count": self.token_count,
            "seq": self.seq,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            id=d["id"],
            doc_id=d["doc_id"],
            page=d["page"],
            section_path=d["section_path"],
            chunk_type=d["chunk_type"],
            content=d["content"],
            token_count=d["token_count"],
            seq=d.get("seq", 0),
            parent_id=d.get("parent_id"),
        )


def _new_id() -> str:
    return uuid.uuid4().hex


def _overlap_tokens(chunk_size: int) -> int:
    """相邻块保留 10% 重叠。"""
    return max(1, int(chunk_size * 0.1))


def build_chunks(
    doc_id: str,
    layout: LayoutResult,
    sections: list[Section],
    doc_title: str = "",
) -> list[Chunk]:
    """多粒度分块：正文按段落/语句切叶子块，表格独立成块。"""
    settings = get_settings()
    chunks: list[Chunk] = []
    seq = 0

    for page in layout.pages:
        section_path = section_path_for_page(sections, page.page_no) or doc_title or "文档正文"

        # 1) 表格：不可分割单元
        for t in page.tables:
            text = t.to_text()
            if not text.strip():
                continue
            token_count = count_tokens(text)
            if token_count <= settings.table_max_tokens:
                seq += 1
                chunks.append(
                    Chunk(
                        id=_new_id(), doc_id=doc_id, page=page.page_no,
                        section_path=section_path, chunk_type="table",
                        content=text, token_count=token_count, seq=seq,
                    )
                )
            else:
                # 超长表格：摘要+分页（阶段一简化为按行分页，共享父块语义）
                pages = split_text_by_tokens(text, settings.table_max_tokens)
                for part in pages:
                    seq += 1
                    chunks.append(
                        Chunk(
                            id=_new_id(), doc_id=doc_id, page=page.page_no,
                            section_path=section_path, chunk_type="table",
                            content=part, token_count=count_tokens(part), seq=seq,
                        )
                    )

        # 2) 正文：按段落累积到目标 token 数
        for para in page.text.split("\n"):
            para = para.strip()
            if not para:
                continue
            para_tokens = count_tokens(para)
            if para_tokens <= settings.chunk_max_tokens:
                seq += 1
                chunks.append(
                    Chunk(
                        id=_new_id(), doc_id=doc_id, page=page.page_no,
                        section_path=section_path, chunk_type="text",
                        content=para, token_count=para_tokens, seq=seq,
                    )
                )
            else:
                # 超长段落：按句子切分为多块，块间 10% 重叠
                seq += 1
                for i, seg in enumerate(_split_paragraph(para, settings)):
                    chunk = Chunk(
                        id=_new_id(), doc_id=doc_id, page=page.page_no,
                        section_path=section_path, chunk_type="text",
                        content=seg, token_count=count_tokens(seg), seq=seq,
                    )
                    if i > 0:
                        chunk.parent_id = chunk.id  # 同一段落片段的父链标记（自引用简化）
                    chunks.append(chunk)
                    seq += 1

    return chunks


def _split_paragraph(para: str, settings) -> list[str]:
    """句子级切分：合并句子至 max，保留 overlap 重叠。"""
    # 按中文句号/分号/换行切句子
    import re

    sentences = re.split(r"(?<=[。；;!?！？])", para)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        sentences = [para]

    target = settings.chunk_target_tokens
    overlap = _overlap_tokens(target)
    parts: list[str] = []
    current = ""
    for s in sentences:
        if current and count_tokens(current + s) > target:
            parts.append(current)
            # 保留尾部 overlap_tokens 的字符作为下一块开头（近似 10% 重叠）
            keep_chars = _chars_for_tokens(current, overlap)
            current = keep_chars + s if keep_chars else s
        else:
            current += s
    if current:
        parts.append(current)
    return parts


def _chars_for_tokens(text: str, n_tokens: int) -> str:
    """取 text 尾部近似 n_tokens 的字符（按 4 字符/token 估拉丁、1 字/token 估中文，简化取整）。"""
    if n_tokens <= 0:
        return ""
    n_chars = min(len(text), n_tokens * 4)
    return text[-n_chars:]
