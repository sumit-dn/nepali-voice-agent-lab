"""Piper voices (VITS/ONNX, real-time on CPU).

Licensing is per voice (see its MODEL_CARD); the piper-tts engine itself is GPL-3.0-or-later.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from voice_lab.providers.base import ProviderError, TTSProvider


class PiperTTS(TTSProvider):
    def _load(self) -> None:
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice

        file = self.spec.params["file"]
        try:
            model = hf_hub_download(str(self.spec.checkpoint), file, revision=self.spec.revision)
            hf_hub_download(str(self.spec.checkpoint), file + ".json", revision=self.spec.revision)
        except (OSError, ValueError) as e:
            raise ProviderError(
                f"Piper voice {file} is not downloaded. "
                f"Run: voice-lab download {self.spec.id}  (size: {self.spec.size})"
            ) from e
        self.voice = PiperVoice.load(model, use_cuda=self.device.startswith("cuda"))

    def _speaker(self, voice: str | None) -> int | None:
        if voice in (None, ""):
            return None
        mapping = self.voice.config.speaker_id_map or {}
        return int(mapping[voice]) if voice in mapping else int(voice)

    def _synthesize(self, text: str, voice: str | None) -> tuple[np.ndarray, int, dict[str, Any]]:
        from piper import SynthesisConfig

        speaker = self._speaker(voice)
        config = SynthesisConfig(speaker_id=speaker, **self.spec.params.get("synthesis", {}))
        chunks = list(self.voice.synthesize(text, syn_config=config))
        if not chunks:
            raise ProviderError("Piper produced no audio (empty text after phonemisation?)")
        audio = np.concatenate([np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in chunks])
        return audio, int(chunks[0].sample_rate), {"speaker_id": speaker, "sentences": len(chunks)}
