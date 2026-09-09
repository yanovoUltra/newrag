"""Development-only whole-page lexical recall; no Gold or financial ontology."""
from __future__ import annotations

import math
import re
from collections import Counter

from app.parsers.base import ParsedPage

_STOP = set('the a an of in on to from for and or what how by did was were is its using according report annual cite evidence show calculation millions dollars'.split())


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', text.casefold()) if t not in _STOP]


def layout_page_text(page: ParsedPage) -> str:
    """Compose one page's body and same-page tables without mutating it."""
    table_texts: list[str] = []
    seen: set[str] = set()
    body_blocks = {block for block in re.split(r"\n\s*\n", page.text or "") if block}
    for table in page.tables:
        if table.page != page.page_no:
            raise ValueError("TABLE_PAGE_OUTSIDE_SOURCE_SCOPE")
        rendered = table.to_text()
        if not rendered or rendered in seen or rendered in body_blocks:
            continue
        seen.add(rendered)
        table_texts.append(rendered)
    return "\n\n".join(part for part in [page.text, *table_texts] if part)


def recall_pages(question: str, pages: dict[int, str], limit: int = 8) -> list[dict]:
    """Rank all explicitly supplied pages, not only previously retrieved chunks.

    Caller owns PDF SHA, tenant and allowed-page checks. Unreadable text cannot
    be recovered by this lexical path; return it separately in the caller.
    """
    if limit < 1:
        raise ValueError('INVALID_PAGE_LIMIT')
    query = set(tokens(question))
    counts = {p: Counter(tokens(text)) for p, text in pages.items()}
    n = len(counts)
    avg = sum(sum(c.values()) for c in counts.values()) / max(n, 1) or 1
    df = Counter(t for c in counts.values() for t in c)
    scores = []
    for page, count in counts.items():
        length = sum(count.values())
        score = sum(math.log(1 + (n - df[t] + .5) / (df[t] + .5)) *
                    (count[t] * 2.2) / (count[t] + 1.2 * (.25 + .75 * length / avg))
                    for t in query if count[t])
        if score:
            scores.append({'page': page, 'score': score})
    return sorted(scores, key=lambda x: (-x['score'], x['page']))[:limit]


def supplemental_page_pool(question: str, pages: dict[int, str], prior_pages: list[int], limit: int = 8) -> list[int]:
    """Append lexical proposals without displacing any prior authorized page.

    This is the candidate pool, not the final model context. A later selector
    enforces the sending budget. Never silently trim it back to the first pages.
    """
    if any(type(p) is not int or p not in pages for p in prior_pages):
        raise ValueError('PRIOR_PAGE_OUTSIDE_SOURCE_SCOPE')
    return list(dict.fromkeys([*prior_pages, *(x['page'] for x in recall_pages(question, pages, limit))]))
