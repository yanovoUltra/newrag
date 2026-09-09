"""Build an authorization-bound pool of physical PDF page candidates.

This module is deliberately local and deterministic: it does not read files,
call a parser, or make an LLM/network request.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

_CID_MARKER = re.compile(r"\(\s*cid\s*:\s*\d+\s*\)", re.IGNORECASE)


def _value(block: Any, *names: str) -> Any:
    if isinstance(block, Mapping):
        for name in names:
            if name in block:
                return block[name]
        return None
    for name in names:
        if hasattr(block, name):
            return getattr(block, name)
    return None


def _text_status(text: str) -> tuple[bool, Decimal, tuple[str, ...]]:
    if not text:
        return False, Decimal("1"), ("empty_text",)
    bad_controls = sum(
        1 for char in text if ord(char) < 32 and char not in {"\n", "\r", "\t"}
    )
    replacement = text.count("\ufffd")
    bad_ratio = Decimal(bad_controls + replacement) / Decimal(len(text))
    reasons: list[str] = []
    if bad_controls and bad_ratio > Decimal("0.02"):
        reasons.append("control_characters")
    if replacement and bad_ratio > Decimal("0.02"):
        reasons.append("replacement_characters")
    if bad_ratio > Decimal("0.02"):
        reasons.append("unusable_text_ratio")
    return not reasons, bad_ratio, tuple(reasons)


@dataclass(frozen=True)
class PageCandidate:
    """One physical page, ordered by the first retrieved block on that page."""

    page: int
    rank: int
    chunk_refs: tuple[str, ...]
    source_text: str
    requires_visual_selection: bool
    text_quality_ratio: Decimal
    quality_reasons: tuple[str, ...] = ()

    @property
    def page_id(self) -> int:
        return self.page


@dataclass
class PageCandidatePool:
    candidates: tuple[PageCandidate, ...]
    identity_missing_count: int = 0
    rejected_external_count: int = 0
    rejected_invalid_page_count: int = 0

    @property
    def page_ids(self) -> tuple[int, ...]:
        return tuple(candidate.page for candidate in self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)


def build_page_candidate_pool(
    blocks: Iterable[Any], *, doc_id: str, org_id: str
) -> PageCandidatePool:
    """Group authorized retrieved blocks into a complete, rank-ordered page pool.

    Blocks with missing identity are excluded and counted separately. Blocks
    explicitly bound to another tenant/document are rejected and counted.
    """
    grouped: dict[int, dict[str, Any]] = {}
    missing = external = invalid_page = 0
    for rank, block in enumerate(blocks):
        block_doc = _value(block, "doc_id", "document_id")
        block_org = _value(block, "org_id", "tenant_id", "tenant")
        if not str(block_doc or "").strip() or not str(block_org or "").strip():
            missing += 1
            continue
        if str(block_doc) != doc_id or str(block_org) != org_id:
            external += 1
            continue
        page = _value(block, "page", "page_number", "physical_page")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            invalid_page += 1
            continue
        entry = grouped.setdefault(
            page,
            {"rank": rank, "refs": [], "texts": [], "requires_visual": False,
             "bad_units": 0, "text_units": 0, "reasons": []},
        )
        chunk_ref = _value(block, "chunk_id", "id", "chunk_ref", "source_ref")
        if chunk_ref is not None and str(chunk_ref) not in entry["refs"]:
            entry["refs"].append(str(chunk_ref))
        text_value = _value(block, "source_text", "content", "text")
        text = "" if text_value is None else str(text_value)
        usable, ratio, reasons = _text_status(text)
        entry["text_units"] += len(text)
        entry["bad_units"] += ratio * Decimal(len(text))
        entry["requires_visual"] |= not usable or bool(_CID_MARKER.search(text))
        entry["reasons"].extend(reasons)
        if text:
            entry["texts"].append(text)
    candidates: list[PageCandidate] = []
    for page, entry in sorted(grouped.items(), key=lambda item: item[1]["rank"]):
        total = Decimal(entry["text_units"])
        quality_ratio = entry["bad_units"] / total if total else Decimal("1")
        reasons = tuple(dict.fromkeys(entry["reasons"]))
        if any(_CID_MARKER.search(text) for text in entry["texts"]):
            reasons = (*reasons, "explicit_cid_marker")
        candidates.append(
            PageCandidate(
                page=page,
                rank=entry["rank"],
                chunk_refs=tuple(entry["refs"]),
                source_text="\n".join(entry["texts"]),
                requires_visual_selection=entry["requires_visual"],
                text_quality_ratio=quality_ratio,
                quality_reasons=tuple(dict.fromkeys(reasons)),
            )
        )
    return PageCandidatePool(tuple(candidates), missing, external, invalid_page)


def select_page_ids(
    selection: Sequence[Any], pool: PageCandidatePool, max_pages: int
) -> list[int]:
    """Validate model-selected page numbers; invalid/insufficient input yields []."""
    if not isinstance(selection, (list, tuple)):
        return []
    if isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 0:
        return []
    if len(selection) > max_pages:
        return []
    allowed = set(pool.page_ids)
    result: list[int] = []
    for page in selection:
        if isinstance(page, bool) or not isinstance(page, int) or page not in allowed:
            return []
        if page in result:
            return []
        result.append(page)
    return result


__all__ = ["PageCandidate", "PageCandidatePool", "build_page_candidate_pool", "select_page_ids"]
