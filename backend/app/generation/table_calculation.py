"""Opt-in table-answer calculation contract; not wired into production chat.

The model extracts operands with source references. It never supplies the result
used by the application. Numeric computation is deterministic; source semantics
still require verification by the caller, not an arithmetic success flag.
"""
from __future__ import annotations

import re
from decimal import Decimal, localcontext
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_NUMBER = re.compile(r'-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?')


class Operand(BaseModel):
    model_config = ConfigDict(extra='forbid')
    value_text: str = Field(min_length=1, max_length=80)
    unit: str = Field(min_length=1, max_length=80)
    scale: Literal[0, 3, 6, 9] = 0
    entity: str = Field(min_length=1, max_length=200)
    metric: str = Field(min_length=1, max_length=200)
    period: str = Field(min_length=1, max_length=100)
    scope: str = Field(min_length=1, max_length=200)
    source_page: int = Field(strict=True, ge=1)
    quote: str = Field(min_length=1, max_length=3000)


class Calculation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    operation: Literal['difference', 'percent_change']
    before: Operand
    after: Operand


class TableAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    answer: str
    source_pages: list[int]
    evidence_quotes: list[str]
    insufficient_evidence: bool
    calculation: Calculation | None = None  # Explicit, not a forbidden surprise field.


def source_decimal(text: str) -> Decimal:
    """Parse source number syntax only. No currency guessing, suffix or FX scaling."""
    s = text.strip()
    negative = s.startswith('(') and s.endswith(')')
    if negative:
        s = s[1:-1].strip()
    if not _NUMBER.fullmatch(s) or (negative and s.startswith('-')):
        raise ValueError('INVALID_SOURCE_NUMBER')
    number = Decimal(s.replace(',', ''))
    return -number if negative else number


def calculate(calculation: Calculation, source_pages: dict[int, str]) -> dict:
    """Check supplied textual provenance and calculate; never prove financial identity.

For IMAGE evidence a separately source-verified transcription is required. A
model's own quote must not be passed back as the purported trusted page text.
"""
    before, after = calculation.before, calculation.after
    for field in ('entity', 'metric', 'scope', 'unit'):
        if getattr(before, field) != getattr(after, field):
            raise ValueError('INCOMPATIBLE_' + field.upper())
    values = []
    for operand in (before, after):
        page = source_pages.get(operand.source_page)
        if page is None:
            raise ValueError('SOURCE_PAGE_NOT_SUPPLIED')
        quote = ' '.join(operand.quote.split())
        if quote not in ' '.join(page.split()):
            raise ValueError('QUOTE_NOT_IN_SOURCE')
        value = source_decimal(operand.value_text)
        # Token boundaries prevent 100 from being "proved" by 2100. Parentheses
        # are preserved so a negative disclosure cannot validate a positive value.
        raw_tokens = re.findall(r'(?<![\w.,])\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?(?![\w.,])', quote)
        if not any(source_decimal(t) == value for t in raw_tokens
                   if t.count('(') == t.count(')')):
            raise ValueError('VALUE_NOT_IN_SOURCE_QUOTE')
        values.append(value)
    with localcontext() as ctx:
        ctx.prec = 180  # Above maximum bounded operand precision; unaffected by caller context.
        baseline = values[0] * (Decimal(10) ** before.scale)
        current = values[1] * (Decimal(10) ** after.scale)
        if calculation.operation == 'difference':
            output_scale = after.scale
            result = (current - baseline) / (Decimal(10) ** output_scale)
            unit = after.unit
            expression = f'({current} - {baseline}) / 10^{output_scale}'
        else:
            if baseline <= 0:
                raise ValueError('NONPOSITIVE_PERCENT_CHANGE_BASE_REQUIRES_CLARIFICATION')
            ctx.prec = 28  # Explicit rounding for potentially recurring percent quotients.
            result = (current - baseline) / baseline * 100
            output_scale, unit = 0, '%'
            expression = f'({current} - {baseline}) / {baseline} * 100'
    return {'value': format(result, 'f'), 'unit': unit, 'scale': output_scale,
            'expression': expression, 'operation': calculation.operation,
            'operands': calculation.model_dump(), 'extraction_method': 'DERIVE',
            'source_pages': sorted({before.source_page, after.source_page}),
            'semantic_identity_verified': False}
