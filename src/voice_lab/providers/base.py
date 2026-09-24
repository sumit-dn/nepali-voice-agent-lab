"""Provider interfaces and result types. Benchmarks and the pipeline only talk to these."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from voice_lab.audio.io import Audio, save_wav
from voice_lab.audio.preprocess import ensure_model_input
from voice_lab.config import ModelSpec


class ProviderError(RuntimeError):
    """Model/server missing or failed. Message tells the user how to fix it."""


@dataclass
class ASRResult:
    text: str
    model: str
    provider: str
    latency_seconds: float
    audio_duration_seconds: float
    real_time_factor: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    latency_seconds: float
    time_to_first_token: float | None
    input_tokens: int | None
    output_tokens: int | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{"name": ..., "arguments": {...}}]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TTSResult:
    audio_path: str  # evaluation copy: WAV, mono, 16-bit PCM, model's native rate
    original_path: str  # model output exactly as produced
    model: str
    voice: str | None
    provider: str
    latency_seconds: float
    audio_duration_seconds: float
    real_time_factor: float
    sample_rate: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VADResult:
    segments: list[tuple[float, float]]  # (start_s, end_s) of detected speech
    model: str
    provider: str
    latency_seconds: float
    audio_duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def speech_seconds(self) -> float:
        return sum(end - start for start, end in self.segments)


def resolve_device(device: str | None) -> str:
    if device not in (None, "auto"):
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Provider(ABC):  # noqa: B024 - subclasses declare the abstract methods
    def __init__(self, spec: ModelSpec, device: str | None = None):
        self.spec = spec
        self.device = resolve_device(device or spec.params.get("device"))
        self.load_seconds: float | None = None

    def load(self) -> None:
        """Load weights / connect once; timed separately from inference."""
        if self.load_seconds is None:
            start = time.perf_counter()
            self._load()
            self.load_seconds = time.perf_counter() - start

    def _load(self) -> None:  # noqa: B027 - optional hook
        pass


class ASRProvider(Provider):
    sample_rate = 16000  # rate the model consumes

    def transcribe(self, audio: Audio) -> ASRResult:
        self.load()
        prepared, notes = ensure_model_input(audio, self.sample_rate)
        start = time.perf_counter()
        text, meta = self._transcribe(prepared)
        latency = time.perf_counter() - start
        if notes:
            meta["model_input_conversion"] = notes
        return ASRResult(
            text=text.strip(),
            model=self.spec.id,
            provider=self.spec.provider,
            latency_seconds=latency,
            audio_duration_seconds=audio.duration,
            real_time_factor=latency / max(audio.duration, 1e-9),
            metadata=meta,
        )

    @abstractmethod
    def _transcribe(self, audio: Audio) -> tuple[str, dict[str, Any]]: ...


class LLMProvider(Provider):
    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult: ...

    def health(self) -> dict[str, Any]:
        return {"ok": True}


class TTSProvider(Provider):
    def synthesize(self, text: str, out_path: str | Path, voice: str | None = None) -> TTSResult:
        self.load()
        voice = voice or self.spec.params.get("voice")
        start = time.perf_counter()
        samples, sr, meta = self._synthesize(text, voice)
        latency = time.perf_counter() - start
        out_path = Path(out_path).with_suffix(".wav")
        original = out_path.with_suffix(".orig.wav")
        save_wav(original, samples, sr, subtype="FLOAT" if samples.dtype.kind == "f" else "PCM_16")
        as_float = samples.astype(np.float32) / (32768.0 if samples.dtype == np.int16 else 1.0)
        mono = as_float.mean(axis=1) if as_float.ndim == 2 else as_float
        save_wav(out_path, np.clip(mono, -1.0, 1.0), sr, subtype="PCM_16")
        duration = len(mono) / sr
        return TTSResult(
            audio_path=str(out_path),
            original_path=str(original),
            model=self.spec.id,
            voice=voice,
            provider=self.spec.provider,
            latency_seconds=latency,
            audio_duration_seconds=duration,
            real_time_factor=latency / max(duration, 1e-9),
            sample_rate=sr,
            metadata=meta,
        )

    @abstractmethod
    def _synthesize(self, text: str, voice: str | None) -> tuple[np.ndarray, int, dict[str, Any]]: ...


class VADProvider(Provider):
    sample_rate = 16000

    def detect_speech(self, audio: Audio) -> VADResult:
        self.load()
        prepared, notes = ensure_model_input(audio, self.sample_rate)
        start = time.perf_counter()
        segments, meta = self._detect(prepared)
        latency = time.perf_counter() - start
        if notes:
            meta["model_input_conversion"] = notes
        return VADResult(segments, self.spec.id, self.spec.provider, latency, audio.duration, meta)

    @abstractmethod
    def _detect(self, audio: Audio) -> tuple[list[tuple[float, float]], dict[str, Any]]: ...
