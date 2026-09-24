from voice_lab.evaluation.llm_checks import run_checks
from voice_lab.evaluation.metrics import (
    align,
    char_errors,
    edit_distance,
    entity_hits,
    mixed_language_hits,
    score_asr,
    word_errors,
)
from voice_lab.evaluation.text import devanagari_ratio, normalize


def test_normalize_keeps_matras_and_maps_digits():
    # Regression guard: a naive [^\w\s] strip deletes Devanagari vowel signs.
    assert normalize("मेरो नाम, राम हो।") == "मेरो नाम राम हो"
    assert normalize("रु. २५,०००") == "रु 25000"
    assert normalize("Hello  WORLD!") == "hello world"
    assert normalize("श्रीमान्‌को") == "श्रीमान्को"


def test_edit_distance_and_alignment():
    assert edit_distance(list("kitten"), list("sitting")) == 3
    assert edit_distance([], list("abc")) == 3
    ops = align(["a", "b", "c", "d"], ["a", "x", "c"])
    assert [o[0] for o in ops] == ["eq", "sub", "eq", "del"]


def test_word_and_char_errors():
    counts, _ = word_errors("मेरो नाम राम हो", "मेरो नाम श्याम हो")
    assert counts == {"sub": 1, "del": 0, "ins": 0, "ref_words": 4}
    assert char_errors("abc", "abc") == {"char_errors": 0, "ref_chars": 3}
    perfect, _ = word_errors("मेरो फोन ९८४१", "मेरो फोन 9841")  # script of digits ignored
    assert perfect["sub"] == 0


def test_entity_hits_numeric_and_text():
    hyp = "मेरो फोन नम्बर 98 41 23 45 67 हो र नाम राम बहादुर थापा"
    hits = entity_hits({"phone_numbers": ["९८४१२३४५६७"], "names": ["राम बहादुर थापा"], "otps": ["12"]}, hyp)
    assert hits["phone_numbers"] == [1, 1]
    assert hits["names"] == [1, 1]
    assert hits["otps"] == [0, 1]  # "12" must not match inside "9841234567"


def test_mixed_language_and_tags():
    assert mixed_language_hits("मलाई tomorrow को appointment", "मलाई टुमोरो को appointment") == [1, 2]
    assert mixed_language_hits("मेरो नाम", "मेरो नाम") is None
    result = score_asr({"text": "मलाई tomorrow को appointment", "entities": {}}, "मलाई टुमोरो को appointment")
    assert "code_switching" in result["tags"] and "english_word" in result["tags"]
    assert result["entities"]["mixed_language"] == [1, 2]


def test_hindi_leak_heuristic():
    from voice_lab.evaluation.llm_checks import hindi_markers

    assert hindi_markers("नहीं, शनिबार बन्द छ।") == ["नहीं"]
    assert hindi_markers("होइन, शनिबार बन्द छ। तपाईंको balance रु. २५,००० छ।") == []


def test_llm_checks():
    calls = [{"name": "get_balance", "arguments": {"customer_id": "c-1001"}}]
    checks = run_checks(
        {"tool": "get_balance", "tool_args": {"customer_id": "C-1001"}, "contains": ["25000"]},
        "तपाईंको balance रु. २५,००० छ।",
        calls,
    )
    assert all(checks.values()), checks
    assert run_checks({"tool_args": {}, "tool": "x"}, "", []) == {"tool_name": False}
    assert run_checks({"max_devanagari": 0.2}, "नमस्ते", []) == {"max_devanagari": False}
    assert devanagari_ratio("abc") == 0.0
