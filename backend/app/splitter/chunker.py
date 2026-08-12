"""Small-to-Big 切分：叶子按目标 token 聚合（含噪声过滤/重叠），表格不可分割（超长摘要+分页），章节级父块。"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from app.core.config import get_settings
from app.parsers.base import LayoutResult, ParsedTable
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
    table_headers: list[str] | None = None  # 表格块元数据：表头（结构化，供过滤/展示）
    n_rows: int | None = None
    n_cols: int | None = None

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
            "table_headers": self.table_headers,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
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
            table_headers=d.get("table_headers"),
            n_rows=d.get("n_rows"),
            n_cols=d.get("n_cols"),
        )


def _new_id() -> str:
    return uuid.uuid4().hex


def _overlap_tokens(chunk_size: int) -> int:
    """相邻块保留 10% 重叠。"""
    return max(1, int(chunk_size * 0.1))


def _table_header_block(t: ParsedTable) -> str:
    """分页块的表头前缀：表头 + 规模统计行（保证每个分页块独立可读）。"""
    lines: list[str] = []
    if t.headers:
        lines.append(" | ".join(str(h) for h in t.headers))
    n_cols = len(t.headers) if t.headers else (len(t.rows[0]) if t.rows else 0)
    lines.append(f"[本表共 {len(t.rows)} 行 × {n_cols} 列，以下为分页明细]")
    return "\n".join(lines)


def _table_summary(t: ParsedTable) -> str:
    """超长表格轻量摘要：表头 + 前 2 行 + 规模统计（不调 LLM）。"""
    lines: list[str] = []
    if t.headers:
        lines.append(" | ".join(str(h) for h in t.headers))
    for r in t.rows[:2]:
        lines.append(" | ".join("" if c is None else str(c) for c in r))
    n_cols = len(t.headers) if t.headers else (len(t.rows[0]) if t.rows else 0)
    lines.append(f"[本表共 {len(t.rows)} 行 × {n_cols} 列，以下为分页明细]")
    return "\n".join(lines)


def _table_meta(t: ParsedTable | None) -> tuple[list[str] | None, int | None, int | None]:
    """表格结构化元数据 (headers, n_rows, n_cols)，供 payload 过滤/展示。"""
    if t is None:
        return None, None, None
    headers = [str(h) for h in t.headers] if t.headers else None
    n_cols = len(t.headers) if t.headers else (len(t.rows[0]) if t.rows else 0)
    return headers, len(t.rows), n_cols


def embedding_text_for(chunk: Chunk, doc_title: str = "") -> str:
    """表格/章节父块的嵌入文本增强：拼接语义锚点（文档标题/公司名 + 章节路径）。

    数字密集的"主要财务指标"表，原始内容只有表头+数值，嵌入向量与语义问句（"某公司某年
    某指标是多少"）相似度低；章节父块正文 2~4K token，标题词占比极低被正文稀释，dense
    检索召不回相关父块（§18 父块路教训、P2-1）——前缀 section_path 让查询的实体+主题词
    在向量空间与标题对齐。仅对**嵌入文本**（dense/sparse 向量）增强，
    展示/上下文仍用 chunk.content 原文。
    """
    if chunk.chunk_type not in ("table", "section"):
        return chunk.content
    anchors: list[str] = []
    if doc_title:
        anchors.append(doc_title)
    if chunk.section_path and chunk.section_path != "文档正文":
        anchors.append(chunk.section_path)
    if not anchors:
        return chunk.content
    return " | ".join(anchors) + "\n" + chunk.content


def build_leaves(
    doc_id: str,
    layout: LayoutResult,
    sections: list[Section],
    doc_title: str = "",
) -> list[Chunk]:
    """收集叶子块：正文按目标 token 聚合（10% 重叠）+ 表格（摘要/分页）。

    语义精切（嵌入相似度断句）在 build_leaves 之后、父块构建之前执行。
    """
    settings = get_settings()
    leaves: list[Chunk] = []
    seq = 0

    def make(
        chunk_type: str,
        content: str,
        page_no: int,
        section_path: str,
        parent_id: str | None = None,
        table: ParsedTable | None = None,
    ) -> Chunk:
        nonlocal seq
        seq += 1
        headers, n_rows, n_cols = _table_meta(table)
        return Chunk(
            id=_new_id(), doc_id=doc_id, page=page_no,
            section_path=section_path, chunk_type=chunk_type,
            content=content, token_count=count_tokens(content), seq=seq,
            parent_id=parent_id,
            table_headers=headers, n_rows=n_rows, n_cols=n_cols,
        )

    for page in layout.pages:
        section_path = section_path_for_page(sections, page.page_no) or doc_title or "文档正文"

        # 1a) 表格：不可分割单元；超长表格 → 摘要 + 分页（分页块重复表头，保证独立可读）
        for t in page.tables:
            text = t.to_text()
            if not text.strip():
                continue
            token_count = count_tokens(text)
            if token_count <= settings.table_max_tokens:
                leaves.append(make("table", text, page.page_no, section_path, table=t))
            else:
                leaves.append(make("table", _table_summary(t), page.page_no, section_path, table=t))
                header_block = _table_header_block(t)
                # 每块容量扣除表头前缀占用，保证加上表头后仍不超限
                capacity = max(1, settings.table_max_tokens - count_tokens(header_block))
                for part in split_text_by_tokens(text, capacity):
                    leaves.append(make("table", f"{header_block}\n\n{part}", page.page_no, section_path, table=t))

        # 1b) 正文：连续行累积到目标 token 数成块，相邻块保留尾部 10% 重叠
        text_lines = [
            line.strip()
            for line in page.text.split("\n")
            if line.strip() and not _is_noise_line(line)
        ]
        if not text_lines:
            continue
        overlap = _overlap_tokens(settings.chunk_target_tokens)
        buffer = ""
        for line in text_lines:
            line_tokens = count_tokens(line)
            if line_tokens > settings.chunk_max_tokens:
                # 超长行（无换行的巨段）：独立按句子切分
                if buffer:
                    leaves.append(make("text", buffer, page.page_no, section_path))
                    buffer = ""
                for seg in _split_paragraph(line, settings):
                    leaves.append(make("text", seg, page.page_no, section_path))
                continue
            if buffer and count_tokens(f"{buffer}\n{line}") > settings.chunk_target_tokens:
                leaves.append(make("text", buffer, page.page_no, section_path))
                # 下一块以本块尾部 overlap 字符开头，消解块边界截断
                keep = _chars_for_tokens(buffer, overlap)
                buffer = f"{keep}\n{line}" if keep else line
            else:
                buffer = f"{buffer}\n{line}" if buffer else line
        if buffer:
            leaves.append(make("text", buffer, page.page_no, section_path))

    return leaves


def build_parent_blocks(doc_id: str, leaves: list[Chunk]) -> list[Chunk]:
    """章节级父块：按 section_path 连续段分组，累积到 parent_target 封块。

    父块 id 会被写进叶子 parent_id，检索侧按 id 回查父块上下文。
    """
    settings = get_settings()
    parent_blocks: list[Chunk] = []

    def flush_segment(segment: list[Chunk]) -> None:
        if not segment:
            return
        path = segment[0].section_path
        page0 = segment[0].page
        parent_id = _new_id()
        buf: list[str] = []
        buf_tokens = 0

        def emit() -> None:
            nonlocal parent_id, buf, buf_tokens
            if not buf:
                return
            content = "\n\n".join(buf)
            parent_blocks.append(
                Chunk(
                    id=parent_id, doc_id=doc_id, page=page0,
                    section_path=path, chunk_type="section",
                    content=content, token_count=count_tokens(content),
                )
            )
            parent_id = _new_id()
            buf = []
            buf_tokens = 0

        for leaf in segment:
            if buf and buf_tokens + leaf.token_count > settings.chunk_parent_target_tokens:
                emit()
            buf.append(leaf.content)
            buf_tokens += leaf.token_count
            leaf.parent_id = parent_id
        emit()

    segment: list[Chunk] = []
    last_path: str | None = None
    for leaf in leaves:
        if last_path is not None and leaf.section_path != last_path:
            flush_segment(segment)
            segment = []
        last_path = leaf.section_path
        segment.append(leaf)
    flush_segment(segment)

    return parent_blocks


def build_chunks(
    doc_id: str,
    layout: LayoutResult,
    sections: list[Section],
    doc_title: str = "",
) -> list[Chunk]:
    """多粒度分块：叶子（正文聚合+重叠 / 表格摘要+分页）+ 章节级父块（2~4K token）。

    返回顺序：父块在前、叶子在后；叶子的 parent_id 指向其所属章节父块。
    """
    leaves = build_leaves(doc_id, layout, sections, doc_title)
    parent_blocks = build_parent_blocks(doc_id, leaves)
    return parent_blocks + leaves


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
