"""XTTS adapter plumbing without the real model: sentence splitting and clear failure messages."""

import sys

import pytest

from voice_lab import config
from voice_lab.config import ModelSpec
from voice_lab.providers.base import ProviderError
from voice_lab.tts import xtts, xtts_worker


def spec(**params):
    return ModelSpec(id="x", provider="xtts", checkpoint="org/xtts", params={"subfolder": "epoch-1", **params})


def test_split_sentences_on_danda_and_punctuation():
    assert xtts_worker.split_sentences("नमस्ते। तपाईंलाई कस्तो छ? ठीक छ! ") == ["नमस्ते।", "तपाईंलाई कस्तो छ?", "ठीक छ!"]
    assert xtts_worker.split_sentences("एउटा मात्र वाक्य") == ["एउटा मात्र वाक्य"]


def test_missing_env_says_how_to_build_it(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    with pytest.raises(ProviderError, match="make install-xtts"):
        xtts.XTTS(spec()).load()


def test_worker_errors_surface_as_provider_errors(tmp_path, monkeypatch):
    # Real worker process, run with this interpreter (no coqui-tts here) against a fake checkpoint folder.
    folder = tmp_path / "snap" / "epoch-1"
    folder.mkdir(parents=True)
    (folder / "model.pth").write_bytes(b"")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(tmp_path / "snap"))
    provider = xtts.XTTS(spec(python=sys.executable))
    with pytest.raises(ProviderError, match="XTTS worker: ModuleNotFoundError"):
        provider.load()
    assert (tmp_path / "results" / "xtts-worker.log").read_text()  # traceback kept for debugging


@pytest.mark.integration
def test_real_oshara_xtts(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from voice_lab.providers.registry import build

    try:
        result = build("tts", "xtts-ne-oshara-e20").synthesize("नमस्ते।", tmp_path / "out.wav")
    except ProviderError as e:
        pytest.skip(str(e))
    assert result.sample_rate == 24000 and result.audio_duration_seconds > 0.3
