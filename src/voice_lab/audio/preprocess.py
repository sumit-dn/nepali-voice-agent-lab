"""Explicit preprocessing. Every op is named in configs/benchmark.yaml and recorded with results.

Nothing here runs implicitly. The one automatic conversion - downmix/resample to the rate a model
requires - happens in `ensure_model_input` and is written into each result's metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from voice_lab.audio.io import Audio, load_audio


def mono(a: Audio) -> Audio:
    return a if a.channels == 1 else Audio(a.samples.mean(axis=1).astype(np.float32), a.sample_rate)


def resample(a: Audio, sample_rate: int) -> Audio:
    if a.sample_rate == sample_rate:
        return a
    g = gcd(a.sample_rate, sample_rate)
    out = signal.resample_poly(a.samples, sample_rate // g, a.sample_rate // g, axis=0)
    return Audio(out.astype(np.float32), sample_rate)


def normalize(a: Audio, peak_dbfs: float = -1.0) -> Audio:
    peak = float(np.max(np.abs(a.samples))) if a.samples.size else 0.0
    if peak == 0.0:
        return a
    return Audio((a.samples * (10 ** (peak_dbfs / 20) / peak)).astype(np.float32), a.sample_rate)


def trim_silence(a: Audio, threshold_dbfs: float = -45.0, pad_ms: int = 100, frame_ms: int = 20) -> Audio:
    """Cut leading/trailing frames whose RMS is below an absolute dBFS threshold."""
    x = mono(a).samples
    frame = max(1, a.sample_rate * frame_ms // 1000)
    n = len(x) // frame
    if n == 0:
        return a
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    loud = np.flatnonzero(20 * np.log10(rms) > threshold_dbfs)
    if loud.size == 0:
        return a
    pad = a.sample_rate * pad_ms // 1000
    start = max(0, loud[0] * frame - pad)
    end = min(len(x), (loud[-1] + 1) * frame + pad)
    return Audio(a.samples[start:end], a.sample_rate)


def bandpass(a: Audio, low_hz: float = 300.0, high_hz: float = 3400.0, order: int = 4) -> Audio:
    """Telephone channel band limit (causal filter, like a real line)."""
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=a.sample_rate, output="sos")
    return Audio(signal.sosfilt(sos, a.samples, axis=0).astype(np.float32), a.sample_rate)


def mulaw(a: Audio, mu: int = 255) -> Audio:
    """G.711-style mu-law round trip with 8-bit quantisation."""
    x = np.clip(a.samples, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round((y + 1) / 2 * mu) / mu * 2 - 1
    out = np.sign(y) * np.expm1(np.abs(y) * np.log1p(mu)) / mu
    return Audio(out.astype(np.float32), a.sample_rate)


def add_noise(a: Audio, snr_db: float = 15.0, seed: int = 0, noise_file: str | None = None) -> Audio:
    """Add white noise, or a looped noise/babble recording, at a fixed SNR (seeded => reproducible)."""
    x = a.samples
    if noise_file:
        n = mono(resample(load_audio(noise_file), a.sample_rate)).samples
        n = np.resize(n, x.shape[0])
        if x.ndim == 2:
            n = n[:, None]
    else:
        n = np.random.default_rng(seed).standard_normal(x.shape).astype(np.float32)
    p_signal = float(np.mean(x**2)) or 1e-12
    p_noise = float(np.mean(n**2)) or 1e-12
    scale = np.sqrt(p_signal / (p_noise * 10 ** (snr_db / 10)))
    return Audio((x + scale * n).astype(np.float32), a.sample_rate)


def denoise(a: Audio, **kwargs: Any) -> Audio:
    try:
        import noisereduce  # optional; not a default dependency
    except ImportError as e:
        raise RuntimeError("`denoise` needs the optional package: uv pip install noisereduce") from e
    out = noisereduce.reduce_noise(y=mono(a).samples, sr=a.sample_rate, **kwargs)
    return Audio(out.astype(np.float32), a.sample_rate)


OPS: dict[str, Callable[..., Audio]] = {
    "mono": mono,
    "resample": resample,
    "normalize": normalize,
    "trim_silence": trim_silence,
    "bandpass": bandpass,
    "mulaw": mulaw,
    "add_noise": add_noise,
    "denoise": denoise,
}


def apply(a: Audio, steps: list[dict[str, Any]]) -> Audio:
    """Apply steps like [{"op": "resample", "sample_rate": 8000}, {"op": "mulaw"}] in order."""
    for step in steps:
        kwargs = {k: v for k, v in step.items() if k != "op"}
        if step.get("op") not in OPS:
            raise ValueError(f"Unknown preprocessing op {step.get('op')!r}; available: {sorted(OPS)}")
        a = OPS[step["op"]](a, **kwargs)
    return a


def ensure_model_input(a: Audio, sample_rate: int) -> tuple[Audio, list[str]]:
    """Downmix/resample to what a model requires. Returns the notes to record in metadata."""
    notes = []
    if a.channels != 1:
        notes.append(f"downmix {a.channels}ch->mono")
        a = mono(a)
    if a.sample_rate != sample_rate:
        notes.append(f"resample {a.sample_rate}->{sample_rate} Hz (model input rate)")
        a = resample(a, sample_rate)
    return a, notes


def load_and_prepare(path: str | Path, steps: list[dict[str, Any]]) -> Audio:
    return apply(load_audio(path), steps)
