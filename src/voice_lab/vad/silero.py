"""Silero VAD (MIT). Weights ship inside the pip package, so nothing is downloaded. Needs torch."""

from __future__ import annotations

from typing import Any

from voice_lab.audio.io import Audio
from voice_lab.providers.base import VADProvider


class SileroVAD(VADProvider):
    def __init__(self, spec: Any, device: str | None = None):
        super().__init__(spec, device)
        self.sample_rate = int(spec.params.get("sample_rate", 16000))  # 8000 or 16000

    def _load(self) -> None:
        from silero_vad import load_silero_vad

        self.model = load_silero_vad(onnx=bool(self.spec.params.get("onnx", False)))

    def _detect(self, audio: Audio) -> tuple[list[tuple[float, float]], dict[str, Any]]:
        import torch
        from silero_vad import get_speech_timestamps

        options = self.spec.params.get("options", {})  # threshold, min_speech_duration_ms, min_silence_duration_ms, ...
        stamps = get_speech_timestamps(
            torch.from_numpy(audio.samples), self.model, sampling_rate=audio.sample_rate, return_seconds=True, **options
        )
        return [(float(s["start"]), float(s["end"])) for s in stamps], {"options": options}
