"""ASR metrics: WER/CER with alignment, critical-entity accuracy, and error tagging.

Per-sample counts are stored (not rates) so corpus-level WER/CER = sum(errors) / sum(reference length).
CER counts Unicode code points of the normalised text (spaces included), so a Devanagari matra is one char.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from voice_lab.evaluation.text import is_devanagari, is_latin, join_digit_groups, normalize

Op = tuple[str, str | None, str | None]  # ("eq"|"sub"|"del"|"ins", ref_token, hyp_token)

# Entity types compared as digit strings (formatting/script of digits ignored).
NUMERIC_ENTITIES = {"phone_numbers", "numbers", "otps", "account_numbers", "currency"}


def _encode(ref: Sequence[str], hyp: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    vocab: dict[str, int] = {}
    r = np.array([vocab.setdefault(t, len(vocab)) for t in ref], dtype=np.int64)
    h = np.array([vocab.setdefault(t, len(vocab)) for t in hyp], dtype=np.int64)
    return r, h


def _rows(r: np.ndarray, h: np.ndarray) -> Iterator[np.ndarray]:
    """Rows of the Levenshtein matrix. Insertions within a row are a running min: d[j] = j + min_k<=j(a[k]-k)."""
    j = np.arange(len(h) + 1)
    row = j.copy()
    yield row
    for i, tok in enumerate(r, 1):
        a = np.empty_like(row)
        a[0] = i
        a[1:] = np.minimum(row[:-1] + (h != tok), row[1:] + 1)
        row = np.minimum.accumulate(a - j) + j
        yield row


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    last = np.arange(len(hyp) + 1)
    for last in _rows(*_encode(ref, hyp)):  # noqa: B007 - we want the final row
        pass
    return int(last[-1])


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[Op]:
    r, h = _encode(ref, hyp)
    d = np.stack(list(_rows(r, h)))
    i, j = len(r), len(h)
    ops: list[Op] = []
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            ops.append(("eq" if r[i - 1] == h[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


def word_errors(ref_text: str, hyp_text: str) -> tuple[dict[str, int], list[Op]]:
    ref, hyp = normalize(ref_text).split(), normalize(hyp_text).split()
    ops = align(ref, hyp)
    kinds = Counter(op for op, _, _ in ops)
    return {"sub": kinds["sub"], "del": kinds["del"], "ins": kinds["ins"], "ref_words": len(ref)}, ops


def char_errors(ref_text: str, hyp_text: str) -> dict[str, int]:
    ref, hyp = list(normalize(ref_text)), list(normalize(hyp_text))
    return {"char_errors": edit_distance(ref, hyp), "ref_chars": len(ref)}


def entity_hits(entities: dict[str, list[str]], hyp_text: str) -> dict[str, list[int]]:
    """{entity_type: [hits, total]}. An entity counts if it appears (normalised) in the hypothesis."""
    hyp = normalize(hyp_text)
    hyp_digits = join_digit_groups(hyp)
    out: dict[str, list[int]] = {}
    for kind, values in entities.items():
        hits = 0
        for value in values:
            if kind in NUMERIC_ENTITIES:
                target = "".join(c for c in normalize(value) if c.isdigit())
                hits += bool(target) and re.search(rf"(?<!\d){target}(?!\d)", hyp_digits) is not None
            else:
                target = normalize(value)
                hits += bool(target) and f" {target} " in f" {hyp} "
        out[kind] = [int(hits), len(values)]
    return out


def mixed_language_hits(ref_text: str, hyp_text: str) -> list[int] | None:
    """[hits, total] for Latin-script words in the reference, or None if the reference has none."""
    english = [w for w in normalize(ref_text).split() if is_latin(w)]
    if not english:
        return None
    hyp = Counter(normalize(hyp_text).split())
    return [sum((Counter(english) & hyp).values()), len(english)]


def error_tags(ops: list[Op], entities: dict[str, list[int]]) -> list[str]:
    tags: set[str] = set()
    for op, ref, hyp in ops:
        if op == "eq":
            continue
        tags.add({"sub": "substitution", "del": "deletion", "ins": "insertion"}[op])
        if any(t and any(c.isdigit() for c in t) for t in (ref, hyp)):
            tags.add("number")
        if ref and is_latin(ref):
            tags.add("english_word")
            if hyp and is_devanagari(hyp):
                tags.add("code_switching")  # English word came back transliterated into Devanagari
        if op == "sub" and ref and hyp and is_devanagari(ref) and is_devanagari(hyp):
            tags.add("devanagari")
    tags.update(f"missed:{kind}" for kind, (hits, total) in entities.items() if hits < total)
    return sorted(tags)


def score_asr(sample: dict[str, Any], hypothesis: str) -> dict[str, Any]:
    words, ops = word_errors(sample["text"], hypothesis)
    entities = entity_hits(sample.get("entities") or {}, hypothesis)
    if (mixed := mixed_language_hits(sample["text"], hypothesis)) is not None:
        entities["mixed_language"] = mixed
    return {
        **words,
        **char_errors(sample["text"], hypothesis),
        "entities": entities,
        "tags": error_tags(ops, entities),
        "error_pairs": [[r, h] for op, r, h in ops if op != "eq"][:25],
    }
