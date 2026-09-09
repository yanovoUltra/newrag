"""Load existing page evidence only after registry authorization.

No parsing, indexing, model call or corpus discovery happens here. The caller
must pass a document identity obtained from its authorized retrieval results.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.parsers.base import LayoutResult
from app.retrieval.pdf_page_recall import layout_page_text
from app.store.registry import get_document

_LEVELS = {"public": 0, "internal": 1, "restricted": 2}
_DOCUMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_MAX_LAYOUT_BYTES = 16 * 1024 * 1024


def load_authorized_page_source(
    doc_id: str, org_id: str, user_visibility: str,
) -> tuple[dict[str, Any], dict[int, str]] | None:
    """Return a source identity and complete page texts, or no usable cache.

    Authorization failures raise before opening any layout. Missing, outdated
    or unsupported documents yield None so the caller can retain chunk evidence.
    The SHA labels the registered PDF; layout SHA independently labels the exact
    bytes read. This does not claim a new PDF/parser quality validation.
    """
    if not _DOCUMENT_ID.fullmatch(doc_id) or user_visibility not in _LEVELS:
        raise PermissionError("PAGE_SOURCE_ACCESS_DENIED")
    document = get_document(doc_id)
    if document is None:
        return None
    if (document.org_id != org_id
            or _LEVELS.get(document.visibility, 99) > _LEVELS[user_visibility]):
        raise PermissionError("PAGE_SOURCE_ACCESS_DENIED")
    if (document.status != "indexed" or document.file_type.lower().lstrip(".") != "pdf"
            or re.fullmatch(r"[0-9a-fA-F]{64}", document.sha256 or "") is None):
        return None

    root = Path(get_settings().resolved_pipeline_dir).resolve()
    path = (root / doc_id / "01_layout.json").resolve()
    if not path.is_relative_to(root):
        raise PermissionError("PAGE_SOURCE_ACCESS_DENIED")
    try:
        # Bounded read also protects against a file growing after its size check.
        with path.open("rb") as stream:
            raw = stream.read(_MAX_LAYOUT_BYTES + 1)
    except FileNotFoundError:
        return None
    if len(raw) > _MAX_LAYOUT_BYTES:
        raise ValueError("PAGE_LAYOUT_SIZE_LIMIT")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("pages"), list):
        raise ValueError("INVALID_PAGE_LAYOUT")
    numbers = [page.get("page_no") for page in payload["pages"] if isinstance(page, dict)]
    if (len(numbers) != len(payload["pages"])
            or any(type(page) is not int or page <= 0 for page in numbers)
            or len(numbers) != len(set(numbers))):
        raise ValueError("INVALID_PAGE_IDENTITY")
    layout = LayoutResult.from_dict(payload)
    texts = {page.page_no: layout_page_text(page) for page in layout.pages}
    if not any(text.strip() for text in texts.values()):
        return None
    return {
        "doc_id": document.id,
        "org_id": document.org_id,
        "doc_name": document.filename,
        "visibility": document.visibility,
        "pdf_sha256": document.sha256.lower(),
        "layout_sha256": hashlib.sha256(raw).hexdigest(),
    }, texts
