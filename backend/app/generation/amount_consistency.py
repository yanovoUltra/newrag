"""Narrow, opt-in check of adjacent monetary restatements in answer text.

This detects contradictory display scales, not source correctness. Unmatched
syntax, currencies, claims and periods are NOT verified. No text is rewritten.
"""
from __future__ import annotations

import re
from decimal import Decimal, localcontext

_NUMBER = r'-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
_SCALE = r'万亿|千亿|百亿|十亿|亿|千万|百万|十万|万|千|百'
_CURRENCY = r'美元|人民币|欧元|日元|港元|英镑'
_PAIR = re.compile(
    rf'(?<![A-Za-z0-9.,+\-])(?P<a>{_NUMBER})\s*(?P<sa>{_SCALE})?\s*'
    rf'(?P<ca>{_CURRENCY})(?:\*\*|__)?\s*[（(]\s*(?:即\s*)?'
    rf'\$?\s*(?P<b>{_NUMBER})\s*(?P<sb>{_SCALE})?\s*'
    rf'(?P<cb>{_CURRENCY})\s*[）)]'
)
_EXPONENT = {'': 0, '百': 2, '千': 3, '万': 4, '十万': 5, '百万': 6,
             '千万': 7, '亿': 8, '十亿': 9, '百亿': 10, '千亿': 11, '万亿': 12}


def check_amount_restatements(answer: str) -> dict:
    """Compare same-currency ``amount (amount)`` pairs with rounding tolerance.

    The tolerance is half the least precise displayed increment. Thus a rounded
    billion figure may restate a precise million figure without a false alarm.
    Different currencies are skipped: no exchange rate or currency inference.
    """
    checked = 0
    conflicts = []
    for match in _PAIR.finditer(answer):
        if match['ca'] != match['cb']:
            continue
        # Bound precision and work; excessively long syntax remains unchecked.
        if max(len(match['a']), len(match['b'])) > 80:
            continue
        with localcontext() as context:
            context.prec = 180
            a, b = (Decimal(match[key].replace(',', '')) for key in ('a', 'b'))
            sa, sb = (_EXPONENT[match[key] or ''] for key in ('sa', 'sb'))
            scaled_a, scaled_b = a.scaleb(sa), b.scaleb(sb)
            increment_a = Decimal(1).scaleb(a.as_tuple().exponent + sa)
            increment_b = Decimal(1).scaleb(b.as_tuple().exponent + sb)
            tolerance = max(increment_a, increment_b) / 2
            checked += 1
            if abs(scaled_a - scaled_b) > tolerance:
                conflicts.append({'start': match.start(), 'end': match.end(),
                                  'code': 'CONTRADICTORY_MONETARY_RESTATEMENT'})
    return {'status': 'CONFLICT' if conflicts else 'NO_CONFLICT_IN_CHECKED_PAIRS' if checked else 'NOT_CHECKED',
            'checked_pairs': checked, 'conflicts': conflicts,
            'source_correctness_verified': False}
