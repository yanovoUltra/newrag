"""Complete-page evidence at the real answer boundary, without global patches."""
from __future__ import annotations

import asyncio

from app.retrieval.page_context import select_complete_page_context
from app.retrieval.page_source import load_authorized_page_source


async def complete_page_evidence(
    question: str, blocks: list[dict], *, org_id: str, visibility: str,
    llm, max_chars: int, max_pages: int = 3,
) -> tuple[list[dict], dict]:
    """Keep all document groups or retain original chunks; never hide one issuer.

    At most three document groups are opened, with one shared character/page
    budget. Unavailable or malformed sources and selector failures retain the
    existing evidence. A registry authorization denial propagates fail-closed.
    No parsing, indexing, Gold access or PDF upload is performed here.
    """
    groups: dict[str, list[dict]] = {}
    for block in blocks:
        doc_id = block.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            return blocks, {"used": False, "reason": "SOURCE_IDENTITY_MISSING"}
        groups.setdefault(doc_id, []).append(block)
    if not groups or len(groups) > max_pages or max_chars <= 0:
        return blocks, {"used": False, "reason": "DOCUMENT_OR_CONTEXT_BUDGET"}
    selected: list[dict] = []
    used_chars = 0
    diagnostics = []
    for index, (doc_id, hits) in enumerate(groups.items()):
        remaining_docs = len(groups) - index
        char_share = (max_chars - used_chars) // remaining_docs
        page_share = (max_pages - len(selected)) // remaining_docs
        try:
            loaded = await asyncio.to_thread(load_authorized_page_source, doc_id, org_id, visibility)
            if loaded is None:
                return blocks, {"used": False, "reason": "PAGE_SOURCE_UNAVAILABLE"}
            source, texts = loaded
            pages, diagnostic = await select_complete_page_context(
                question, hits, source=source, page_texts=texts, llm=llm,
                max_pages=page_share, total_chars=char_share,
                selector_input_chars=96000 // len(groups),
            )
        except PermissionError:
            raise
        except Exception:
            # No raw exception, path, question or provider response is exposed.
            return blocks, {"used": False, "reason": "PAGE_SELECTION_UNAVAILABLE"}
        if not diagnostic["sent_pages"]:
            return blocks, {"used": False, "reason": diagnostic["fallback_reason"]}
        for page in pages:
            page["visibility"] = source["visibility"]
        selected.extend(pages)
        used_chars += sum(len(page["content"]) for page in pages)
        diagnostics.append({"page_count": len(pages), "chars": diagnostic["budget"]["sent_chars"]})
    return selected, {"used": True, "reason": None, "page_count": len(selected),
                      "content_chars": used_chars, "documents": diagnostics}
