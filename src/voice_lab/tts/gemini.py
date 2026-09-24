"""OPTIONAL cloud baseline: Gemini TTS (raw PCM s16le, mono). Requires --allow-cloud."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from voice_lab.providers import gemini
from voice_lab.providers.base import ProviderError, TTSProvider


class GeminiTTS(TTSProvider):
    def _load(self) -> None:
        gemini.api_key()

    def _synthesize(self, text: str, voice: str | None) -> tuple[np.ndarray, int, dict[str, Any]]:
        body = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice or "Kore"}}},
            },
        }
        # Google documents occasional 500s from preview TTS models ("text tokens instead of audio"); retry once.
        try:
            response = gemini.generate_content(str(self.spec.checkpoint), body)
        except ProviderError:
            response = gemini.generate_content(str(self.spec.checkpoint), body)
        blob = next((p["inlineData"] for p in gemini.parts(response) if "inlineData" in p), None)
        if blob is None:
            raise ProviderError("Gemini TTS returned no audio part")
        pcm = np.frombuffer(base64.b64decode(blob["data"]), dtype=np.int16)
        return (
            pcm,
            int(self.spec.params.get("sample_rate", 24000)),
            {"mime_type": blob.get("mimeType"), "external": True},
        )
