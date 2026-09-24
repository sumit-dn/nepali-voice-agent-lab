"""Maps `provider:` names in configs/*.yaml to adapter classes (imported lazily so heavy deps stay optional)."""

from __future__ import annotations

import logging
import sys
from importlib import import_module
from typing import Any

from voice_lab.config import ModelSpec, get_model, load_models
from voice_lab.providers.base import Provider, ProviderError

log = logging.getLogger(__name__)

ADAPTERS: dict[str, Any] = {
    "hf-asr": "voice_lab.asr.hf_asr:HFASR",
    "indicconformer": "voice_lab.asr.indicconformer:IndicConformerASR",
    "gemini-asr": "voice_lab.asr.gemini:GeminiASR",
    "openai-compat": "voice_lab.llm.openai_compat:OpenAICompatLLM",
    "piper": "voice_lab.tts.piper:PiperTTS",
    "parler": "voice_lab.tts.parler:ParlerTTS",
    "xtts": "voice_lab.tts.xtts:XTTS",
    "gemini-tts": "voice_lab.tts.gemini:GeminiTTS",
    "silero": "voice_lab.vad.silero:SileroVAD",
}

INSTALL_HINT = "uv sync --extra local --extra cpu   (or --extra gpu on a CUDA machine)"
_SENDS = {"asr": "audio", "llm": "text", "tts": "text", "vad": "audio"}


class CloudNotAllowed(ProviderError):
    pass


def adapter_class(provider: str) -> type[Provider]:
    if provider == "none":
        raise ProviderError(
            "No adapter implemented for this candidate yet (registry entry kept for licensing/tracking)."
        )
    target = ADAPTERS.get(provider)
    if target is None:
        raise ProviderError(f"No adapter for provider {provider!r}; known: {sorted(ADAPTERS)}")
    if isinstance(target, type):
        return target
    module, _, name = target.partition(":")
    try:
        return getattr(import_module(module), name)
    except ImportError as e:
        raise ProviderError(
            f"Provider {provider!r} needs packages that are not installed ({e}). Install: {INSTALL_HINT}"
        ) from e


def build(
    kind: str,
    model_id: str,
    *,
    device: str | None = None,
    allow_cloud: bool = False,
    spec: ModelSpec | None = None,
) -> Provider:
    spec = spec or get_model(kind, model_id)
    if not spec.enabled:
        log.warning("%s is disabled in configs/%s.yaml; running it because it was requested explicitly", spec.id, kind)
    if spec.external:
        service = spec.params.get("service", "an external cloud API")
        if not allow_cloud:
            raise CloudNotAllowed(
                f"{spec.id} sends {_SENDS[kind]} to {service}. Nothing was sent. "
                "Re-run with --allow-cloud to confirm (use a personal/test API key, never production)."
            )
        print(f"WARNING: This experiment sends {_SENDS[kind]} to {service} ({spec.id}).", file=sys.stderr)
    return adapter_class(spec.provider)(spec, device=device)


def select(kind: str, model_ids: list[str] | None) -> list[ModelSpec]:
    """Explicit ids, or every enabled model of that kind."""
    models = load_models(kind)
    if model_ids:
        return [get_model(kind, m) for m in model_ids]
    return [m for m in models.values() if m.enabled]
