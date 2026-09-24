import numpy as np
import pytest

from voice_lab.audio.io import Audio, load_audio, save_wav
from voice_lab.audio.preprocess import add_noise, apply, ensure_model_input, mulaw, resample, trim_silence


def sine(sr: int = 16000, seconds: float = 1.0) -> Audio:
    t = np.arange(int(sr * seconds)) / sr
    return Audio((0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)


def test_resample_and_mono():
    a = resample(sine(16000), 8000)
    assert a.sample_rate == 8000 and len(a.samples) == 8000
    stereo = Audio(np.stack([sine().samples, sine().samples], axis=1), 16000)
    out = apply(stereo, [{"op": "mono"}])
    assert out.channels == 1


def test_mulaw_roundtrip_is_close():
    a = sine()
    assert np.max(np.abs(mulaw(a).samples - a.samples)) < 0.02


def test_add_noise_hits_target_snr_and_is_seeded():
    a = sine()
    noisy = add_noise(a, snr_db=10, seed=1)
    noise = noisy.samples - a.samples
    snr = 10 * np.log10(np.mean(a.samples**2) / np.mean(noise**2))
    assert abs(snr - 10) < 0.1
    assert np.array_equal(noisy.samples, add_noise(a, snr_db=10, seed=1).samples)


def test_trim_silence():
    sr = 16000
    x = np.concatenate([np.zeros(sr), sine(sr, 0.5).samples, np.zeros(sr)]).astype(np.float32)
    trimmed = trim_silence(Audio(x, sr), pad_ms=0)
    assert 0.45 < trimmed.duration < 0.6


def test_ensure_model_input_records_conversion():
    out, notes = ensure_model_input(sine(8000), 16000)
    assert out.sample_rate == 16000 and notes == ["resample 8000->16000 Hz (model input rate)"]
    assert ensure_model_input(sine(16000), 16000)[1] == []


def test_unknown_op_and_format(tmp_path):
    with pytest.raises(ValueError):
        apply(sine(), [{"op": "magic"}])
    with pytest.raises(ValueError):
        load_audio(tmp_path / "x.aac")
    path = save_wav(tmp_path / "x.wav", sine().samples, 16000)
    assert load_audio(path).sample_rate == 16000
