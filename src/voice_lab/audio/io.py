"""Audio loading/saving. Decoding converts to float32 and nothing else."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}  # via libsndfile >= 1.1 (bundled in soundfile wheels)


@dataclass
class Audio:
    samples: np.ndarray  # float32, shape (frames,) or (frames, channels), range [-1, 1]
    sample_rate: int

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else self.samples.shape[1]

    @property
    def duration(self) -> float:
        return self.samples.shape[0] / self.sample_rate


def load_audio(path: str | Path) -> Audio:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"{path}: unsupported format {path.suffix!r}; supported: {sorted(SUPPORTED_EXTENSIONS)}")
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return Audio(data[:, 0] if data.shape[1] == 1 else data, int(sr))


def peak_dbfs(audio: Audio) -> float:
    """Peak level in dBFS; -inf for digital silence (e.g. an unplugged or muted microphone)."""
    peak = float(np.max(np.abs(audio.samples))) if audio.samples.size else 0.0
    return 20 * float(np.log10(peak)) if peak > 0 else float("-inf")


def save_wav(path: str | Path, samples: np.ndarray, sample_rate: int, subtype: str = "PCM_16") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype=subtype)
    return path
