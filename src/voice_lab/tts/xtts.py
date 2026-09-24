"""Coqui XTTS-v2 checkpoints (e.g. Oshara/xtts-v2-nepali), run in a worker process inside .venv-xtts.

coqui-tts 0.27 imports APIs that transformers 5 removed, and with torch>=2.9 it requires torchcodec + FFmpeg, so
it cannot share the lab's main env. `make install-xtts` builds .venv-xtts (transformers<5, torch 2.8) and this
adapter drives tts/xtts_worker.py there over JSON lines; measured latency includes a few ms of IPC.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from voice_lab import config
from voice_lab.providers.base import ProviderError, TTSProvider

WORKER = Path(__file__).with_name("xtts_worker.py")


class XTTS(TTSProvider):
    proc: subprocess.Popen[str] | None = None

    def _load(self) -> None:
        from huggingface_hub import snapshot_download

        p = self.spec.params
        python = config.ROOT / p.get("python", ".venv-xtts/bin/python")
        if not python.exists():
            raise ProviderError(
                f"XTTS needs its own environment ({python} not found). Build it once: make install-xtts"
            )
        sub = p["subfolder"]
        hint = f"Download it first: voice-lab download {self.spec.id}  (size: {self.spec.size})"
        try:
            folder = (
                Path(
                    snapshot_download(
                        str(self.spec.checkpoint), revision=self.spec.revision, allow_patterns=[f"{sub}/*"]
                    )
                )
                / sub
            )
        except (OSError, ValueError) as e:
            raise ProviderError(f"{self.spec.checkpoint}/{sub} is not downloaded. {hint}") from e
        if not (folder / "model.pth").exists():
            raise ProviderError(f"{self.spec.checkpoint}/{sub}/model.pth is not downloaded. {hint}")
        self.log_path = config.RESULTS_DIR / "xtts-worker.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # one request at a time over the pipe
        with self.log_path.open("a") as log:  # the child keeps its own handle
            self.proc = subprocess.Popen(
                [str(python), str(WORKER)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
            )
        info = self._call({"cmd": "load", "dir": str(folder), "device": self.device})
        self.sample_rate_out, self.speakers = int(info["sample_rate"]), info["speakers"]

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        assert self.proc is not None and self.proc.stdin is not None and self.proc.stdout is not None
        with self._lock:
            self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        if not line:
            raise ProviderError(f"XTTS worker exited (code {self.proc.poll()}); details in {self.log_path}")
        reply: dict[str, Any] = json.loads(line)
        if not reply["ok"]:
            raise ProviderError(f"XTTS worker: {reply['error']} (traceback in {self.log_path})")
        return reply

    def _synthesize(self, text: str, voice: str | None) -> tuple[np.ndarray, int, dict[str, Any]]:
        p = self.spec.params
        voice = voice or p.get("voice", "Claribel Dervla")
        if Path(voice).is_file():
            voice = str(Path(voice).resolve())  # reference clip for cloning; the worker may run elsewhere
        fd, out = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            info = self._call(
                {
                    "cmd": "synth",
                    "text": text,
                    "voice": voice,
                    "language": p.get("language", "hi"),
                    "out": out,
                    "options": p.get("options", {}),
                }
            )
            audio, sr = sf.read(out, dtype="float32")
        finally:
            os.unlink(out)
        cloned = voice not in self.speakers
        return (
            audio,
            int(sr),
            {
                "voice": "reference clip" if cloned else voice,
                "cloned": cloned,
                "sentences": info["sentences"],
                "language_token": p.get("language", "hi"),
            },
        )

    def __del__(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()  # also exits on its own when stdin closes
