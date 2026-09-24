"""Paths, .env loading, and the YAML model registry (configs/{asr,llm,tts,vad}.yaml)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("VOICE_LAB_ROOT") or Path(__file__).resolve().parents[2])
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
KINDS = ("asr", "llm", "tts", "vad")

# Every credential is optional: self-hosted experiments need none of them.
OPTIONAL_ENV = {
    "GEMINI_API_KEY": "Gemini cloud baselines (models with `external: true`); use a personal/test key",
    "HF_TOKEN": "gated Hugging Face checkpoints (e.g. Gemma)",
    "OLLAMA_BASE_URL": "Ollama server (default http://localhost:11434)",
    "VLLM_BASE_URL": "vLLM server (default http://localhost:8000/v1)",
}


class ConfigError(ValueError):
    """Bad or missing configuration; the message says what to fix."""


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader: KEY=VALUE lines. Never overrides variables already set.

    Empty values (`HF_TOKEN=` from the template) are skipped: exporting them would mask a value filled in
    later for every subprocess started by an already-running process (e.g. downloads from the UI).
    """
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value := value.strip().strip("'\""):
            os.environ.setdefault(key.strip(), value)


def missing_credentials() -> dict[str, str]:
    return {key: why for key, why in OPTIONAL_ENV.items() if not os.environ.get(key)}


@dataclass
class ModelSpec:
    """One registry entry. Provider-specific options live in `params`."""

    id: str
    provider: str
    kind: str = ""
    name: str = ""
    checkpoint: str | None = None
    revision: str | None = None
    enabled: bool = True
    external: bool = False  # True = sends data to a third-party service; needs --allow-cloud
    languages: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    license: dict[str, Any] = field(default_factory=dict)
    size: str | None = None
    hardware: str | None = None
    verified: str | None = None
    sources: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def commercial_use(self) -> str:
        return str(self.license.get("commercial_use", "REVIEW REQUIRED"))

    def fingerprint(self) -> str:
        """Hash of everything that changes model output; used to reuse cached results."""
        key = json.dumps([self.provider, self.checkpoint, self.revision, self.params], sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:12]


_FIELDS = {f.name for f in dataclasses.fields(ModelSpec)}


def load_models(kind: str, config_dir: Path | None = None) -> dict[str, ModelSpec]:
    if kind not in KINDS:
        raise ConfigError(f"Unknown model kind {kind!r}; expected one of {KINDS}")
    path = (config_dir or CONFIG_DIR) / f"{kind}.yaml"
    if not path.exists():
        raise ConfigError(f"Missing registry file {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models: dict[str, ModelSpec] = {}
    for raw in data.get("models") or []:
        where = f"{path.name} model {raw.get('id')!r}"
        if unknown := set(raw) - _FIELDS:
            raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
        if not raw.get("id") or not raw.get("provider"):
            raise ConfigError(f"{where}: `id` and `provider` are required")
        if raw["id"] in models:
            raise ConfigError(f"{where}: duplicate id")
        models[raw["id"]] = ModelSpec(**{**raw, "kind": kind})
    return models


def get_model(kind: str, model_id: str, config_dir: Path | None = None) -> ModelSpec:
    models = load_models(kind, config_dir)
    if model_id not in models:
        raise ConfigError(f"Unknown {kind} model {model_id!r}. Available: {', '.join(models) or '(none)'}")
    return models[model_id]


def load_benchmark_config(config_dir: Path | None = None) -> dict[str, Any]:
    path = (config_dir or CONFIG_DIR) / "benchmark.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
