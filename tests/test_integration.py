"""Real-model smoke tests. Skipped by default; run with `pytest -m integration` after downloading
the small models: `voice-lab download piper-ne-google-x-low` (28 MB). Silero ships with its package."""

import pytest

from voice_lab.audio.io import load_audio
from voice_lab.providers.base import ProviderError
from voice_lab.providers.registry import build

pytestmark = pytest.mark.integration


def test_piper_then_silero(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    try:
        tts = build("tts", "piper-ne-google-x-low")
        result = tts.synthesize("नमस्ते, तपाईंलाई कस्तो छ?", tmp_path / "out.wav")
    except ProviderError as e:
        pytest.skip(str(e))
    assert result.sample_rate == 16000 and result.audio_duration_seconds > 0.5
    vad = build("vad", "silero-vad").detect_speech(load_audio(result.audio_path))
    assert vad.segments, "Silero should find speech in synthesized audio"
