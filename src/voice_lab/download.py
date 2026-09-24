"""Explicit model downloads (the only command allowed to fetch weights) and local-availability checks."""

from __future__ import annotations

import os
import shutil
import subprocess
from fnmatch import fnmatch
from functools import cache
from pathlib import Path
from typing import Any

from voice_lab.config import ModelSpec
from voice_lab.providers.base import ProviderError

HF_PROVIDERS = {"hf-asr", "indicconformer", "piper", "parler", "xtts"}
# Transformers checkpoints: weights + tokenizer/config only (skip TF/Flax/ONNX duplicates).
HF_ASR_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"]


def allow_patterns(spec: ModelSpec) -> list[str] | None:
    if spec.params.get("allow_patterns"):
        return list(spec.params["allow_patterns"])
    if spec.provider == "hf-asr":
        return HF_ASR_PATTERNS
    if spec.provider == "xtts":
        return [f"{spec.params['subfolder']}/*"]  # one epoch folder, not the whole repo
    if spec.provider == "piper":
        file = spec.params["file"]
        return [file, file + ".json", str(Path(file).parent / "MODEL_CARD")]
    return None  # whole repo (remote-code models need every asset)


def _hf_cache_dir(repo: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"


def refresh_status() -> None:
    """Forget cached Ollama model lists (e.g. after `ollama pull` while the UI is running)."""
    _ollama_models.cache_clear()


@cache
def _ollama_models(root: str) -> set[str] | None:
    from voice_lab.utils.http import get_json

    try:
        return {m["name"] for m in get_json(f"{root}/api/tags", timeout=3).get("models", [])}
    except ProviderError:
        return None


def local_status(spec: ModelSpec) -> str:
    if spec.provider == "none":
        return "no adapter"
    if spec.external:
        return "cloud (opt-in)"
    if spec.provider == "silero":
        return "bundled with pip pkg"
    if spec.provider == "openai-compat":
        if spec.params.get("server") != "ollama":
            return f"server ({spec.params.get('server', '?')})"
        root = os.environ.get(spec.params.get("base_url_env", ""), "") or spec.params.get("default_base_url", "")
        pulled = _ollama_models(root.rstrip("/"))
        if pulled is None:
            return "ollama unreachable"
        return "pulled" if spec.checkpoint in pulled else "not pulled"
    if spec.provider in HF_PROVIDERS and spec.checkpoint:
        snapshot = f"snapshots/{spec.revision or '*'}"  # pinned revision must be the one on disk
        wanted = {
            "piper": f"{snapshot}/{spec.params.get('file')}",
            "xtts": f"{snapshot}/{spec.params.get('subfolder')}/model.pth",
        }.get(spec.provider, f"{snapshot}/*")
        return "downloaded" if any(_hf_cache_dir(spec.checkpoint).glob(wanted)) else "not downloaded"
    return "unknown"


def hf_plan(spec: ModelSpec) -> dict[str, Any]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(str(spec.checkpoint), revision=spec.revision, files_metadata=True)
    patterns = allow_patterns(spec)
    files = [s for s in info.siblings or [] if patterns is None or any(fnmatch(s.rfilename, p) for p in patterns)]
    if spec.provider == "hf-asr" and not any(f.rfilename.endswith(".safetensors") for f in files):
        patterns = [*(patterns or []), "pytorch_model*.bin"]  # older fine-tunes ship only .bin weights
        files = [s for s in info.siblings or [] if any(fnmatch(s.rfilename, p) for p in patterns)]
    return {"patterns": patterns, "files": len(files), "bytes": sum(s.size or 0 for s in files), "gated": info.gated}


def check_access(repo: str) -> None:
    """Gated repos need accepted terms + a token; say how, before any byte is downloaded."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError

    try:
        HfApi().auth_check(repo)
    except GatedRepoError as e:
        token_note = "" if os.environ.get("HF_TOKEN") else "\n(HF_TOKEN is not set right now.)"
        raise ProviderError(
            f"No access to gated model {repo}. One-time fix:\n"
            f"  1. Log in at https://huggingface.co and open https://huggingface.co/{repo}\n"
            "  2. Click 'Agree and access repository' (fill the form if asked)\n"
            "  3. Create a READ token at https://huggingface.co/settings/tokens (personal, not a production one)\n"
            "  4. Put it in .env as HF_TOKEN=hf_...  then run the download again" + token_note
        ) from e


def download(spec: ModelSpec, yes: bool = False) -> None:
    print(f"Model:     {spec.id} ({spec.kind}, provider {spec.provider})")
    print(f"Checkpoint {spec.checkpoint} @ {spec.revision or 'latest'}")
    print(f"License:   {spec.license.get('id', '?')} | commercial use: {spec.commercial_use}")
    print(f"Hardware:  {spec.hardware or 'not documented'}")
    if spec.external or spec.provider in ("none", "silero"):
        print("Nothing to download for this entry.")
        return
    if spec.provider == "openai-compat":
        if spec.params.get("server") != "ollama":
            raise ProviderError(
                f"{spec.id} is served by {spec.params.get('server')}; start that server with the model."
            )
        print(f"Size:      {spec.size} (Ollama)")
        if not yes and input(f"Run `ollama pull {spec.checkpoint}`? [y/N] ").strip().lower() != "y":
            print("Cancelled.")
            return
        if not shutil.which("ollama"):
            raise ProviderError("`ollama` CLI not found. Install Ollama first: https://ollama.com/download")
        subprocess.run(["ollama", "pull", str(spec.checkpoint)], check=True)
        return
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    try:
        plan = hf_plan(spec)
        print(f"Size:      {plan['bytes'] / 1e9:.2f} GB in {plan['files']} files (registry says: {spec.size})")
        if plan["gated"]:
            print(f"Gated:     yes ({plan['gated']}) - checking that you have access…")
            check_access(str(spec.checkpoint))
            print("Access:    OK")
        free = shutil.disk_usage(Path.home()).free
        if plan["bytes"] > free * 0.9:
            raise ProviderError(f"Not enough disk: need {plan['bytes'] / 1e9:.1f} GB, free {free / 1e9:.1f} GB")
        if not yes and input("Download now? [y/N] ").strip().lower() != "y":
            print("Cancelled.")
            return
        path = snapshot_download(str(spec.checkpoint), revision=spec.revision, allow_patterns=plan["patterns"])
    except HfHubHTTPError as e:
        raise ProviderError(f"Hugging Face request failed for {spec.checkpoint}: {e}") from e
    print(f"Downloaded to {path}")
