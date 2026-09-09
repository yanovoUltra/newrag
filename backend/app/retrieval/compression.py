"""Traceable extractive context compression for narrative and table chunks."""

from __future__ import annotations

import re
from typing import Any

_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?|[^\n]+$")
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
_NUMBER_RE = re.compile(r"\d[\d,.%万亿元亿美元港币人民币]*")


def _bigrams(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.casefold())
    return {normalized[index : index + 2] for index in range(max(0, len(normalized) - 1))}


def _query_features(question: str) -> tuple[set[str], set[str]]:
    exact = {token.casefold() for token in _ASCII_WORD_RE.findall(question)}
    exact.update(_NUMBER_RE.findall(question))
    return _bigrams(question), exact


def _score(text: str, query_bigrams: set[str], exact: set[str]) -> float:
    text_lower = text.casefold()
    overlap = len(_bigrams(text) & query_bigrams)
    exact_hits = sum(1 for token in exact if token and token in text_lower)
    return overlap + exact_hits * 4.0


def _spans(content: str, *, table: bool) -> list[tuple[int, int, str]]:
    if table:
        spans = []
        cursor = 0
        for line in content.splitlines(keepends=True):
            clean = line.rstrip("\r\n")
            end = cursor + len(clean)
            if clean.strip():
                spans.append((cursor, end, clean))
            cursor += len(line)
        return spans
    return [
        (match.start(), match.end(), match.group(0).strip())
        for match in _SENTENCE_RE.finditer(content)
        if match.group(0).strip()
    ]


def compress_text(
    question: str,
    content: str,
    *,
    max_chars: int,
    table: bool = False,
) -> tuple[str, list[dict[str, int]]]:
    """Return exact source excerpts and their original character offsets."""

    if len(content) <= max_chars:
        return content, [{"start": 0, "end": len(content)}]
    units = _spans(content, table=table)
    if not units:
        return content[:max_chars], [{"start": 0, "end": min(len(content), max_chars)}]
    query_bigrams, exact = _query_features(question)
    ranked = sorted(
        range(len(units)),
        key=lambda index: (_score(units[index][2], query_bigrams, exact), -index),
        reverse=True,
    )
    selected: set[int] = {0} if table else set()
    for index in ranked:
        selected.add(index)
        if not table:
            if index > 0:
                selected.add(index - 1)
            if index + 1 < len(units):
                selected.add(index + 1)
        ordered = sorted(selected)
        if sum(len(units[item][2]) + 1 for item in ordered) >= max_chars:
            break
    excerpts: list[str] = []
    offsets: list[dict[str, int]] = []
    used = 0
    for index in sorted(selected):
        start, end, text = units[index]
        remaining = max_chars - used
        if remaining <= 0:
            break
        excerpt = text[:remaining]
        excerpts.append(excerpt)
        offsets.append({"start": start, "end": start + len(excerpt)})
        used += len(excerpt) + 1
    separator = "\n" if table else ""
    return separator.join(excerpts), offsets


def compress_context(
    question: str,
    blocks: list[dict[str, Any]],
    *,
    block_max_chars: int,
    total_max_chars: int,
) -> list[dict[str, Any]]:
    if not blocks:
        return []
    per_block = max(400, min(block_max_chars, total_max_chars // len(blocks)))
    output = []
    for block in blocks:
        original = str(block.get("content") or "")
        content, spans = compress_text(
            question,
            original,
            max_chars=per_block,
            table=block.get("chunk_type") == "table",
        )
        item = {
            **block,
            "content": content,
            "compression_spans": spans,
            "compressed_from_chars": len(original),
        }
        parent = str(block.get("parent_content") or "")
        if parent:
            parent_text, parent_spans = compress_text(
                question,
                parent,
                max_chars=per_block,
                table=False,
            )
            item["parent_content"] = parent_text
            item["parent_compression_spans"] = parent_spans
        output.append(item)
    return output
