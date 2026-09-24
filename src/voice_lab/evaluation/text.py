"""Text normalisation shared by ASR scoring and LLM validators. Always applied to both sides equally."""

from __future__ import annotations

import re
import unicodedata

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_DIGIT_GROUP_COMMA = re.compile(r"(?<=\d),(?=\d)")
_SPACED_DIGITS = re.compile(r"(?<=\d) (?=\d)")

# Recorded with every ASR result so scores are comparable across runs.
NORMALIZATION = (
    "NFC; drop ZWJ/ZWNJ; Devanagari digits->ASCII; drop digit-group commas; "
    "Unicode punctuation+symbols (incl. danda) -> space; lowercase; collapse whitespace"
)


def normalize(text: str) -> str:
    # NB: not `[^\w\s]` - Python's \w excludes Devanagari vowel signs/virama (Mn/Mc), which would destroy words.
    text = unicodedata.normalize("NFC", text).replace("‌", "").replace("‍", "")
    text = _DIGIT_GROUP_COMMA.sub("", text.translate(DEVANAGARI_DIGITS))
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.lower().split())


def join_digit_groups(normalized: str) -> str:
    """'98 41 234' -> '9841234' so spoken-in-groups numbers compare as one number."""
    return _SPACED_DIGITS.sub("", normalized)


def is_latin(word: str) -> bool:
    return any("a" <= c <= "z" or "A" <= c <= "Z" for c in word)


def is_devanagari(word: str) -> bool:
    return any("ऀ" <= c <= "ॿ" for c in word)


def devanagari_ratio(text: str) -> float:
    """Share of Devanagari among letters (incl. combining marks); 0.0 when there are no letters."""
    letters = [c for c in text if c.isalpha() or unicodedata.category(c).startswith("M")]
    return sum("ऀ" <= c <= "ॿ" for c in letters) / len(letters) if letters else 0.0
