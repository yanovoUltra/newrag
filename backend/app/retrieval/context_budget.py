"""Opt-in narrative packing experiment; production compression is unchanged."""
import re

from app.retrieval.compression import _query_features, _score


def pack_narrative(question: str, leaf: str, parent: str, budget: int) -> dict:
    """Choose complete source sentences from leaf AND parent under one budget.

    Never guess a continuation, use an answer locator, or cut a numeric token.
    Oversize sentences are skipped, not silently truncated. Offsets refer to the
    original leaf/parent, so omitted or reordered spans remain auditable.
    """
    if budget <= 0:
        raise ValueError('INVALID_BUDGET')
    features = _query_features(question)
    candidates = []
    seen = set()
    for origin, text in (('content', leaf), ('parent_content', parent)):
        for match in re.finditer(r'.+?(?:[.!?。！？](?=\s|$)|\Z)', text, re.S):
            raw = match.group()
            start = match.start() + len(raw) - len(raw.lstrip())
            excerpt = raw.strip()
            key = re.sub(r'\s+', '', excerpt)
            if not key or key in seen or len(excerpt) > budget:
                continue
            seen.add(key)
            candidates.append((_score(excerpt, *features), origin, start, excerpt))
    chosen, used = [], 0
    for score, origin, start, excerpt in sorted(candidates, key=lambda c: (-c[0], c[1], c[2])):
        cost = len(excerpt) + bool(chosen)
        if used + cost <= budget:
            chosen.append((origin, start, excerpt))
            used += cost
    chosen.sort(key=lambda c: (c[0], c[1]))
    return {'content': '\n'.join(c[2] for c in chosen), 'parent_content': '',
            'source_spans': [{'origin': origin, 'start': start, 'end': start + len(text)}
                             for origin, start, text in chosen]}
