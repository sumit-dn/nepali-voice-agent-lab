"""Gemini ASR response parsing, with the HTTP call mocked (no network, no key needed)."""

import numpy as np
import pytest

from voice_lab.asr.gemini import GeminiASR
from voice_lab.audio.io import Audio
from voice_lab.config import ModelSpec
from voice_lab.providers import gemini
from voice_lab.providers.base import ProviderError


def transcribe_with(monkeypatch, parts):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(gemini, "generate_content", lambda model, body: {"candidates": [{"content": {"parts": parts}}]})
    asr = GeminiASR(ModelSpec(id="g", provider="gemini-asr", checkpoint="gemini-x", external=True))
    return asr.transcribe(Audio(np.full(16000, 0.1, dtype=np.float32), 16000)).text


def test_transcribe_model_shape(monkeypatch):  # real gemini-3.5-transcribe response, 2026-09-24
    assert transcribe_with(monkeypatch, [{"audioTranscription": {"text": "मेरो नाम राम हो।"}}]) == "मेरो नाम राम हो।"


def test_general_model_shape(monkeypatch):
    assert transcribe_with(monkeypatch, [{"text": "नमस्ते"}, {"text": " हजुर"}]) == "नमस्ते हजुर"


def test_unknown_shape_is_an_error_not_an_empty_transcript(monkeypatch):
    with pytest.raises(ProviderError, match="no transcript field"):
        transcribe_with(monkeypatch, [{"somethingNew": {"words": []}}])
