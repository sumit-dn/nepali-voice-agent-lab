"""Isolated lab fixture: temp configs/results + fake providers (no model downloads, no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from voice_lab import config
from voice_lab.audio.io import Audio
from voice_lab.providers import registry
from voice_lab.providers.base import ASRProvider, LLMProvider, LLMResult, TTSProvider, VADProvider

LICENSE = {"id": "MIT", "commercial_use": "yes"}


class FakeASR(ASRProvider):
    def _transcribe(self, audio: Audio) -> tuple[str, dict[str, Any]]:
        return self.spec.params.get("reply", "मेरो फोन नम्बर ९८४१२३४५६७ हो"), {"fake": True}


class FakeLLM(LLMProvider):
    def generate(
        self, messages: list[dict[str, Any]], *, tools: Any = None, temperature: Any = None, max_tokens: Any = None
    ) -> LLMResult:
        return LLMResult(
            text="नमस्ते, हजुर।",
            model=self.spec.id,
            provider=self.spec.provider,
            latency_seconds=0.01,
            time_to_first_token=0.005,
            input_tokens=10,
            output_tokens=4,
        )


class FakeTTS(TTSProvider):
    def _synthesize(self, text: str, voice: str | None) -> tuple[np.ndarray, int, dict[str, Any]]:
        t = np.arange(16000) / 16000
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000, {"voice": voice}


class FakeVAD(VADProvider):
    def _detect(self, audio: Audio) -> tuple[list[tuple[float, float]], dict[str, Any]]:
        return [(0.1, max(0.2, audio.duration - 0.1))], {}


def tone(path: Path, seconds: float = 1.0, sr: int = 16000) -> Path:
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
    return path


@pytest.fixture
def lab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import yaml

    cfg = tmp_path / "configs"
    cfg.mkdir()
    (tmp_path / "data").mkdir()
    tone(tmp_path / "data" / "a.wav")
    tone(tmp_path / "data" / "b.wav", sr=8000)

    def write(name: str, models: list[dict[str, Any]]) -> None:
        (cfg / name).write_text(yaml.safe_dump({"models": models}, allow_unicode=True), encoding="utf-8")

    write(
        "asr.yaml",
        [
            {"id": "fake-asr", "provider": "fake-asr", "license": LICENSE},
            {"id": "fake-cloud-asr", "provider": "fake-asr", "external": True, "enabled": False, "license": LICENSE},
            {"id": "tracked-only", "provider": "none", "enabled": False},
        ],
    )
    write("llm.yaml", [{"id": "fake-llm", "provider": "fake-llm", "license": LICENSE}])
    write("tts.yaml", [{"id": "fake-tts", "provider": "fake-tts", "license": LICENSE}])
    write("vad.yaml", [{"id": "fake-vad", "provider": "fake-vad", "license": LICENSE}])
    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir()
    rows = [
        {
            "id": "s1",
            "audio": "data/a.wav",
            "text": "मेरो फोन नम्बर ९८४१२३४५६७ हो",
            "category": "numbers",
            "entities": {"phone_numbers": ["९८४१२३४५६७"]},
        },
        {"id": "s2", "audio": "data/b.wav", "text": "मलाई tomorrow को appointment चाहिन्छ", "category": "mixed"},
    ]
    (manifests / "asr.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (manifests / "llm.jsonl").write_text(
        json.dumps(
            {
                "id": "l1",
                "category": "general",
                "prompt": "नमस्ते",
                "expect": {"contains": ["नमस्ते"]},
                "human_eval": ["nepali_quality"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (manifests / "tts.jsonl").write_text(
        json.dumps({"id": "t1", "category": "basic", "text": "नमस्ते"}, ensure_ascii=False), encoding="utf-8"
    )
    (cfg / "benchmark.yaml").write_text(
        yaml.safe_dump(
            {
                "datasets": {k: f"{manifests}/{k}.jsonl" for k in ("asr", "llm", "tts")},
                "preprocess_profiles": {
                    "original": [],
                    "clean_8k": [{"op": "mono"}, {"op": "resample", "sample_rate": 8000}],
                },
                "asr_profiles": ["original"],
                "telephone_suite": ["original", "clean_8k"],
                "llm": {"system_prompt": "test"},
                "pipeline": {"vad": "fake-vad", "asr": "fake-asr", "llm": "fake-llm", "tts": "fake-tts"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "CONFIG_DIR", cfg)
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setitem(registry.ADAPTERS, "fake-asr", FakeASR)
    monkeypatch.setitem(registry.ADAPTERS, "fake-llm", FakeLLM)
    monkeypatch.setitem(registry.ADAPTERS, "fake-tts", FakeTTS)
    monkeypatch.setitem(registry.ADAPTERS, "fake-vad", FakeVAD)
    return tmp_path
