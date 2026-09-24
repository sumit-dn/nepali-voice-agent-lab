"""OPTIONAL cloud baseline: Gemini transcription with inline WAV audio. Requires --allow-cloud."""

from __future__ import annotations

import base64
import io
from typing import Any

import soundfile as sf

from voice_lab.audio.io import Audio
from voice_lab.providers import gemini
from voice_lab.providers.base import ASRProvider, ProviderError


class GeminiASR(ASRProvider):
    def _load(self) -> None:
        gemini.api_key()  # fail before any audio is read/sent

    def _transcribe(self, audio: Audio) -> tuple[str, dict[str, Any]]:
        buf = io.BytesIO()
        sf.write(buf, audio.samples, audio.sample_rate, format="WAV", subtype="PCM_16")
        content: list[dict[str, Any]] = [
            {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(buf.getvalue()).decode()}}
        ]
        p = self.spec.params
        if p.get("prompt"):
            content.append({"text": p["prompt"]})
        body: dict[str, Any] = {"contents": [{"parts": content}]}
        if p.get("generation_config"):
            body["generationConfig"] = p["generation_config"]
        response = gemini.generate_content(str(self.spec.checkpoint), body)
        parts = gemini.parts(response)
        # General models answer with {"text": ...}; gemini-*-transcribe with {"audioTranscription": {"text": ...}}.
        texts = [
            part["text"] if "text" in part else part["audioTranscription"].get("text", "")
            for part in parts
            if "text" in part or "audioTranscription" in part
        ]
        if parts and not texts:  # an unknown response shape must not look like "heard nothing"
            raise ProviderError(f"Gemini returned no transcript field; part keys: {[sorted(p) for p in parts]}")
        return "".join(texts), {"usage": response.get("usageMetadata"), "external": True}
