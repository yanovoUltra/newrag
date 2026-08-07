"""Token 估算工具：优先 tiktoken，缺失时用字符启发式。"""

from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

_enc = None


def count_tokens(text: str) -> int:
    global _enc
    if _enc is None:
        try:
            import tiktoken

            _enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _enc = False  # type: ignore[assignment]
    if _enc:
        return len(_enc.encode(text))
    # 启发式：CJK 约 1 字/token，拉丁约 4 字符/token
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return cjk + max(1, other // 4)


def split_text_by_tokens(text: str, max_tokens: int) -> list[str]:
    """按 max_tokens 硬切（用于超长表格分页 / 无标点巨段兜底）。"""
    if count_tokens(text) <= max_tokens:
        return [text]
    parts: list[str] = []
    buffer = ""
    for line in text.splitlines(keepends=True):
        if count_tokens(line) > max_tokens:
            # 单行超长（无换行的巨表格行/巨段）：按估算字符硬切，防止整行进缓冲导致超限
            if buffer:
                parts.append(buffer)
                buffer = ""
            parts.extend(_hard_split_line(line, max_tokens))
            continue
        if buffer and count_tokens(buffer + line) > max_tokens:
            parts.append(buffer)
            buffer = line
        else:
            buffer += line
    if buffer:
        parts.append(buffer)
    return parts


def _hard_split_line(line: str, max_tokens: int) -> list[str]:
    """无换行长行按字符硬切：按 1 字符/token 上界取块，保证任意语言不超 max_tokens。"""
    chunk_chars = max(1, max_tokens)
    return [line[i : i + chunk_chars] for i in range(0, len(line), chunk_chars)]
