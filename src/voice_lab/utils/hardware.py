"""Hardware/software inventory, recorded with every benchmark run."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import psutil

from voice_lab import config

PACKAGES = ("torch", "torchaudio", "transformers", "huggingface_hub", "onnxruntime", "piper-tts", "silero-vad")


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _gpus() -> tuple[list[dict[str, Any]], str | None]:
    """GPU list + CUDA version, via torch if installed, else nvidia-smi."""
    if importlib.util.find_spec("torch"):
        import torch

        if torch.cuda.is_available():
            gpus = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1),
                }
                for i in range(torch.cuda.device_count())
            ]
            return gpus, torch.version.cuda
    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        gpus = [
            {"name": n.strip(), "vram_gb": round(float(m) / 1024, 1)}
            for n, m in (ln.split(",") for ln in out.splitlines() if ln)
        ]
        return gpus, None
    return [], None


def system_info() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    gpus, cuda = _gpus()
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(),
        "ram_total_gb": round(vm.total / 1e9, 1),
        "ram_available_gb": round(vm.available / 1e9, 1),
        "disk_free_gb": round(shutil.disk_usage(config.ROOT).free / 1e9, 1),
        "gpus": gpus,
        "cuda": cuda,
        "packages": _versions(),
    }
