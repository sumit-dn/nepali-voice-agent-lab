"""JSONL dataset manifests (see data/README.md for the schema)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from voice_lab import config

REQUIRED = {
    "asr": ("id", "audio", "text"),
    "llm": ("id", "category"),
    "tts": ("id", "text"),
    "pipeline": ("id", "audio"),
}


class ManifestError(ValueError):
    pass


def resolve_audio(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else config.ROOT / p


def load_manifest(path: str | Path, kind: str, check_files: bool = True) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise ManifestError(
            f"Manifest not found: {path}. Format: data/README.md. For a synthetic smoke set: "
            "voice-lab dataset synth --tts piper-ne-google-x-low"
        )
    rows, seen, problems = [], set(), []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"{path.name}:{n}: invalid JSON ({e.msg})")
            continue
        if missing := [k for k in REQUIRED[kind] if k not in row]:
            problems.append(f"{path.name}:{n}: missing {missing}")
            continue
        if row["id"] in seen:
            problems.append(f"{path.name}:{n}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        if kind == "llm" and not ("prompt" in row or "messages" in row):
            problems.append(f"{path.name}:{n}: needs `prompt` or `messages`")
        if check_files and "audio" in row and not resolve_audio(row["audio"]).exists():
            problems.append(f"{path.name}:{n}: audio file not found: {row['audio']}")
        rows.append(row)
    if problems:
        raise ManifestError("\n".join(problems))
    return rows


def manifest_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
