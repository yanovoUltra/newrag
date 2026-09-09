"""Deterministic, source-bound PDF phrase localization.

This module deliberately does not parse PDFs, infer boxes, or call a model.  A
caller must provide either an already-open PyMuPDF page or its extracted word
boxes.  Ambiguous matches fail closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


class CitationLocalizationError(ValueError):
    """Base error for a citation that cannot be safely certified."""


class AmbiguousPhraseError(CitationLocalizationError):
    """The phrase occurs more than once in the supplied page."""


class SourceIdentityError(CitationLocalizationError):
    """The supplied source identity does not satisfy the requested binding."""


@dataclass(frozen=True, slots=True)
class PdfSourceIdentity:
    """Stable identity that must travel with a localized citation.

    ``document_sha256`` is a declared reference; the caller must verify the
    file's digest before invoking this helper.
    """

    document_id: str
    document_sha256: str
    page_number: int

    def __post_init__(self) -> None:
        if not self.document_id or not self.document_sha256 or self.page_number < 1:
            raise ValueError("document_id, document_sha256, and positive page_number are required")


@dataclass(frozen=True, slots=True)
class WordBox:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True, slots=True)
class LocalizedCitation:
    phrase: str
    bbox: tuple[float, float, float, float]
    source: PdfSourceIdentity


def _word_box(item: WordBox | Sequence[Any]) -> WordBox:
    if isinstance(item, WordBox):
        box = item
    else:
        if len(item) < 5:
            raise ValueError("word boxes must contain x0, y0, x1, y1, and text")
        # PyMuPDF's get_text("words") order is x0, y0, x1, y1, text, ...
        box = WordBox(str(item[4]), float(item[0]), float(item[1]), float(item[2]), float(item[3]))
    if not all(math.isfinite(value) for value in (box.x0, box.y0, box.x1, box.y1)):
        raise ValueError("word box coordinates must be finite")
    if box.x1 < box.x0 or box.y1 < box.y0:
        raise ValueError("word box coordinates must not be inverted")
    return box


def localize_phrase(
    phrase: str,
    *,
    source: PdfSourceIdentity,
    words: Iterable[WordBox | Sequence[Any]] | None = None,
    page: Any | None = None,
    expected_source: PdfSourceIdentity | None = None,
    expected_page_number: int | None = None,
) -> LocalizedCitation | None:
    """Locate one exact phrase and return its union bbox plus source identity.

    Matching is case-sensitive and token-exact: ``17`` does not match ``117``
    or ``17.0``.  ``page`` is only used to obtain ``get_text("words")``; no
    model-proposed coordinates are accepted.  No match returns ``None`` and
    multiple matches raise :class:`AmbiguousPhraseError`.
    """
    if not phrase or not phrase.split():
        raise ValueError("phrase must be non-empty")
    if words is not None and page is not None:
        raise ValueError("provide words or page, not both")
    if expected_source is not None and source != expected_source:
        raise SourceIdentityError("source identity mismatch")
    if expected_page_number is not None and source.page_number != expected_page_number:
        raise SourceIdentityError("page identity mismatch")
    if page is not None:
        page_index = getattr(page, "number", None)
        if isinstance(page_index, int) and source.page_number != page_index + 1:
            raise SourceIdentityError("page identity mismatch")
        words = page.get_text("words")
    if words is None:
        raise ValueError("words or an already-open page is required")

    boxes = [_word_box(item) for item in words]
    if page is not None:
        rect = getattr(page, "rect", None)
        if rect is None:
            raise ValueError("page must expose a finite rect for bounded citation")
        bounds = tuple(float(getattr(rect, name)) for name in ("x0", "y0", "x1", "y1"))
        if not all(math.isfinite(value) for value in bounds) or bounds[2] < bounds[0] or bounds[3] < bounds[1]:
            raise ValueError("page rect must be finite and non-inverted")
        if any(
            box.x0 < bounds[0]
            or box.y0 < bounds[1]
            or box.x1 > bounds[2]
            or box.y1 > bounds[3]
            for box in boxes
        ):
            raise ValueError("word box lies outside page bounds")
    target = phrase.split()
    matches: list[tuple[float, float, float, float]] = []
    for start in range(len(boxes) - len(target) + 1):
        window = boxes[start : start + len(target)]
        if [box.text for box in window] != target:
            continue
        matches.append(
            (
                min(box.x0 for box in window),
                min(box.y0 for box in window),
                max(box.x1 for box in window),
                max(box.y1 for box in window),
            )
        )
    if len(matches) > 1:
        raise AmbiguousPhraseError("phrase is not unique on the supplied page")
    if not matches:
        return None
    return LocalizedCitation(phrase=phrase, bbox=matches[0], source=source)
