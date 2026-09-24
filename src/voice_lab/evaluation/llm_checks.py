"""Deterministic validators for LLM test cases. Subjective qualities are flagged for human review instead.

`expect` keys (all optional), see data/manifests/llm.jsonl:
  tool / tool_args     a call to this tool happened (anywhere in the tool loop) with these arguments
  no_tool              no tool was called
  contains / not_contains / contains_any   normalised substring checks (digits compared as digit strings)
  min_devanagari / max_devanagari          share of Devanagari letters in the reply (language/script control)
  max_words            brevity for voice
"""

from __future__ import annotations

import re
from typing import Any

from voice_lab.evaluation.text import devanagari_ratio, join_digit_groups, normalize


def _contains(text_norm: str, needle: str) -> bool:
    n = normalize(needle)
    if n.replace(" ", "").isdigit():
        return re.search(rf"(?<!\d){n.replace(' ', '')}(?!\d)", join_digit_groups(text_norm)) is not None
    return n in text_norm


def _same(actual: Any, expected: Any) -> bool:
    a, e = normalize(str(actual)), normalize(str(expected))
    digits = "".join(c for c in e if c.isdigit())
    if digits and digits == e.replace(" ", ""):
        return "".join(c for c in a if c.isdigit()) == digits
    return a == e


# Whole-word Hindi forms that standard Nepali does not use (Nepali equivalents in comments). A HEURISTIC:
# it flags likely Hindi leakage for human review; it is not a language identifier.
HINDI_MARKERS = {
    "नहीं",  # होइन / छैन
    "है",
    "हैं",  # छ / छन्
    "मैं",  # म
    "आप",
    "आपका",
    "आपको",  # तपाईं
    "क्या",  # के
    "कैसे",  # कसरी
    "लेकिन",  # तर
    "बहुत",  # धेरै
    "रहा",
    "रही",  # दै...छ
    "गया",  # गयो
    "किया",  # गर्यो
}


def hindi_markers(text: str) -> list[str]:
    return sorted({w for w in normalize(text).split() if w in HINDI_MARKERS})


def run_checks(expect: dict[str, Any], text: str, tool_calls: list[dict[str, Any]]) -> dict[str, bool]:
    norm = normalize(text)
    checks: dict[str, bool] = {}
    if tool := expect.get("tool"):
        matching = [c for c in tool_calls if c.get("name") == tool]
        checks["tool_name"] = bool(matching)
        if args := expect.get("tool_args"):
            got = matching[0].get("arguments", {}) if matching else {}
            checks["tool_args"] = isinstance(got, dict) and all(_same(got.get(k), v) for k, v in args.items())
    if expect.get("no_tool"):
        checks["no_tool"] = not tool_calls
    for needle in expect.get("contains", []):
        checks[f"contains:{needle}"] = _contains(norm, needle)
    if options := expect.get("contains_any"):
        checks["contains_any"] = any(_contains(norm, o) for o in options)
    for needle in expect.get("not_contains", []):
        checks[f"not_contains:{needle}"] = not _contains(norm, needle)
    if "min_devanagari" in expect:
        checks["min_devanagari"] = devanagari_ratio(text) >= expect["min_devanagari"]
    if "max_devanagari" in expect:
        checks["max_devanagari"] = devanagari_ratio(text) <= expect["max_devanagari"]
    if "max_words" in expect:
        checks["max_words"] = len(norm.split()) <= expect["max_words"]
    return checks
