"""Experimental source-preserving packing; no topic guessing or Gold lookup."""
from copy import deepcopy

from app.generation.prompts import build_answer_messages
from app.retrieval.context_packing import pack_context, pack_context_deduplicated


def contains(parent, child):
    return bool(child.get('content')) and (
        parent.get('chunk_type') == 'section' and parent.get('chunk_id') == child.get('parent_id')
        and all(parent.get(k) is not None and parent.get(k) == child.get(k)
                for k in ('doc_id', 'org_id', 'visibility'))
        and child['content'] in parent.get('content', ''))


def source_safe_context(blocks, max_chars):
    safe = deepcopy(blocks)
    for block in safe:
        # A page-level inferred title is not proof of this fragment's subject.
        # Keep the metadata for audit, but cite the original page instead.
        block['original_section_path'] = block.get('section_path', '')
        block['section_path'] = '原文'
    admitted = pack_context(safe, max_chars)
    # Passing admitted blocks only deliberately disables the old naive refill.
    selected = pack_context_deduplicated(admitted, max_chars)
    represented = {(b.get('doc_id'), b.get('chunk_id')) for b in admitted}
    for block in safe:
        key = (block.get('doc_id'), block.get('chunk_id'))
        if key in represented or not block.get('content'):
            continue
        if any(contains(p, block) or contains(block, p) for p in selected):
            continue
        block.pop('parent_content', None)
        text = build_answer_messages('', selected + [block])[-1]['content'].split('\n\n【问题】\n')[0]
        if len(text) <= max_chars:
            selected.append(block)
            represented.add(key)
    return selected
