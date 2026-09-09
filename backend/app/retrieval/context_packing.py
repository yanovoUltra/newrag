"""Whole-source admission. No lookup, text synthesis, or Gold-dependent selection."""
from copy import deepcopy

from app.generation.prompts import build_answer_messages


def pack_context(blocks: list[dict], max_chars: int) -> list[dict]:
    selected = []
    seen = set()

    def admit(source):
        key = (source.get('doc_id'), source.get('chunk_id'))
        if key in seen:
            return True
        block = deepcopy(source)
        block.pop('parent_content', None)
        text = build_answer_messages('', selected + [block])[-1]['content']
        knowledge = text.split('\n\n【问题】\n', 1)[0]
        if len(knowledge) > max_chars:
            return False
        selected.append(block)
        seen.add(key)
        return True

    for block in blocks:
        if not block.get('content'):
            continue
        if admit(block) or block.get('chunk_type') != 'section':
            continue
        for child in blocks:
            if (child.get('content') and child.get('parent_id') == block.get('chunk_id')
                    and all(child.get(k) == block.get(k) for k in ('doc_id', 'org_id', 'visibility'))
                    and child['content'] in block['content']):
                admit(child)
    return selected


def pack_context_deduplicated(blocks: list[dict], max_chars: int) -> list[dict]:
    """Compact admitted exact children, preserving source labels within the budget."""
    selected = pack_context(blocks, max_chars)
    original = deepcopy(selected)
    removed = set()
    for child in selected:
        for parent in selected:
            if (parent.get('chunk_type') == 'section'
                    and parent.get('chunk_id') != child.get('chunk_id')
                    and child.get('parent_id') == parent.get('chunk_id')
                    and all(parent.get(k) is not None and parent.get(k) == child.get(k)
                            for k in ('doc_id', 'org_id', 'visibility'))
                    and child.get('page') is not None
                    and parent['content'].count(child['content']) == 1
                    and child.get('chunk_type') != 'section'):
                ref = {k: child.get(k) for k in (
                    'doc_id', 'chunk_id', 'doc_name', 'section_path', 'page', 'chunk_type')}
                ref.update(start=parent['content'].index(child['content']), length=len(child['content']))
                parent.setdefault('contained_citations', []).append(ref)
                removed.add((child.get('doc_id'), child.get('chunk_id')))
                break
    selected = [b for b in selected if (b.get('doc_id'), b.get('chunk_id')) not in removed]

    def fits(items):
        prompt = build_answer_messages('', items)[-1]['content']
        return len(prompt.split('\n\n【问题】\n', 1)[0]) <= max_chars

    # Annotation overhead is real context. Never sacrifice an old source to fit it.
    if not fits(selected):
        return original
    represented = {(b.get('doc_id'), b.get('chunk_id')) for b in original}
    for source in blocks:
        key = (source.get('doc_id'), source.get('chunk_id'))
        if key in represented or not source.get('content'):
            continue
        block = deepcopy(source)
        block.pop('parent_content', None)
        if fits(selected + [block]):
            selected.append(block)
            represented.add(key)
    return selected
