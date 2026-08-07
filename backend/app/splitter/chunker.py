"""Small-to-Big 切分：正文按行聚合到目标 token 数成块（过滤目录/页眉等噪声行），表格不可分割。"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.parsers.base import LayoutResult
from app.parsers.structure import Section, section_path_for_page
from app.splitter.tokens import count_tokens, split_text_by_tokens

# 目录/页眉/页脚等单独一行出现时的常见噪声（精确匹配，保守过滤）
_NOISE_EXACT = {
    "目录", "目 录", "contents", "table of contents", "toc",
    "页码", "page", "本页无正文", "（本页无正文）", "(this page intentionally left blank)",
}


def _is_noise_line(line: str) -> bool:
    """判定是否为碎片噪声行（纯数字页码、单汉字、装饰线、目录占位等）。"""
    s = line.strip()
    if not s:
        return True
    if s.lower() in _NOISE_EXACT:
        return True
    # 纯数字 / 数字+分隔符 / 百分号（如 "1"、"23"、"1/2"、"10.5%"、"第 3 页"）
    if re.fullmatch(r"[\d\s\.,\-—–/\\%:：()（）]+", s):
        return True
    if re.fullmatch(r"第\s*\d+\s*页", s, re.IGNORECASE):
        return True
    if re.fullmatch(r"page\s*\d+\s*(of\s*\d+)?", s, re.IGNORECASE):
        return True
    # 单个汉字（多为目录行，如 "目"、"录"）
    if len(s) == 1 and re.fullmatch(r"[\u4e00-\u9fff]", s):
        return True
    # 装饰线 / 连续符号
    if re.fullmatch(r"[\-—–_=*·~.\s]{2,}", s):
        return True
    return False


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
    """多粒度分块：正文连续行聚合到目标 token 数成块（过滤噪声行），表格不可分割。"""
    settings = get_settings()
    chunks: list[Chunk] = []
    seq = 0

    def add_chunk(
        chunk_type: str,
        content: str,
        page_no: int,
        section_path: str,
        parent_id: str | None = None,
    ) -> None:
        nonlocal seq
        seq += 1
        chunks.append(
            Chunk(
                id=_new_id(), doc_id=doc_id, page=page_no,
                section_path=section_path, chunk_type=chunk_type,
                content=content, token_count=count_tokens(content), seq=seq,
                parent_id=parent_id,
            )
        )

    for page in layout.pages:
        section_path = section_path_for_page(sections, page.page_no) or doc_title or "文档正文"

        # 1) 表格：不可分割单元
        for t in page.tables:
            text = t.to_text()
            if not text.strip():
                continue
            token_count = count_tokens(text)
            if token_count <= settings.table_max_tokens:
                add_chunk("table", text, page.page_no, section_path)
            else:
                # 超长表格：摘要+分页（阶段一简化为按行分页）
                for part in split_text_by_tokens(text, settings.table_max_tokens):
                    add_chunk("table", part, page.page_no, section_path)

        # 2) 正文：连续行累积到目标 token 数成块，过滤目录/页眉/页脚等噪声行
        text_lines = [
            line.strip()
            for line in page.text.split("\n")
            if line.strip() and not _is_noise_line(line)
        ]
        if not text_lines:
            continue
        # 页级聚合父块：同页所有叶子共享 parent_id，检索时按组扩展上下文（阶段二）
        parent_id = _new_id()
        buffer = ""
        for line in text_lines:
            line_tokens = count_tokens(line)
            if line_tokens > settings.chunk_max_tokens:
                # 超长行（无换行的巨段）：独立按句子切分
                if buffer:
                    add_chunk("text", buffer, page.page_no, section_path, parent_id)
                    buffer = ""
                for seg in _split_paragraph(line, settings):
                    add_chunk("text", seg, page.page_no, section_path, parent_id)
                continue
            if buffer and count_tokens(f"{buffer}\n{line}") > settings.chunk_target_tokens:
                add_chunk("text", buffer, page.page_no, section_path, parent_id)
                buffer = line
            else:
                buffer = f"{buffer}\n{line}" if buffer else line
        if buffer:
            add_chunk("text", buffer, page.page_no, section_path, parent_id)

    return chunks


def _split_paragraph(para: str, settings) -> list[str]:
    """句子级切分：合并句子至 target，保留 overlap 重叠；单句超长时硬切兜底。"""
    # 按中文句号/分号/换行切句子
    sentences = re.split(r"(?<=[。；;!?！？])", para)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        sentences = [para]

    target = settings.chunk_target_tokens
    overlap = _overlap_tokens(target)
    parts: list[str] = []
    current = ""
    for s in sentences:
        if count_tokens(s) > settings.chunk_max_tokens:
            # 单句超长（如长串数字/无标点段落）：先清空缓冲，再硬切兜底
            if current:
                parts.append(current)
                current = ""
            parts.extend(split_text_by_tokens(s, settings.chunk_max_tokens))
            continue
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
