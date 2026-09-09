"""Detect explicit decoding damage, without guessing language or financial meaning."""
import re


def has_decoding_damage(text: str) -> bool:
    # CID placeholders are single undecoded glyphs, not nine readable characters.
    text = re.sub(r"\(cid:\d+\)", "\ufffd", text)
    bad = sum(c == "\ufffd" or (ord(c) < 32 and c not in "\n\r\t") for c in text)
    return bad >= 3 and bad / max(1, len(text)) >= 0.01
