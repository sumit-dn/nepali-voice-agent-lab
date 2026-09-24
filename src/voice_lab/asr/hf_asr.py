"""Any transformers ASR checkpoint (Whisper + fine-tunes, wav2vec2/CTC) through the HF pipeline."""

from __future__ import annotations

from typing import Any

from voice_lab.audio.io import Audio
from voice_lab.providers.base import ASRProvider, ProviderError


class HFASR(ASRProvider):
    def _load(self) -> None:
        from transformers import pipeline

        p = self.spec.params
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.spec.checkpoint,
                revision=self.spec.revision,
                device=self.device,
                **p.get("pipeline_kwargs", {}),
            )
        except OSError as e:
            raise ProviderError(
                f"{self.spec.checkpoint} could not be loaded ({e.__class__.__name__}: {str(e)[:200]}). "
                f"Download it explicitly: voice-lab download {self.spec.id}  (size: {self.spec.size})"
            ) from e

    def _transcribe(self, audio: Audio) -> tuple[str, dict[str, Any]]:
        p = self.spec.params
        kwargs: dict[str, Any] = {}
        if p.get("generate_kwargs"):
            kwargs["generate_kwargs"] = p["generate_kwargs"]
        if audio.duration > 30:
            kwargs.update(p.get("long_form_kwargs", {}))
        out: Any = self.pipe({"raw": audio.samples, "sampling_rate": audio.sample_rate}, **kwargs)
        return out["text"], {"checkpoint": self.spec.checkpoint, "device": self.device, "call_kwargs": kwargs}
