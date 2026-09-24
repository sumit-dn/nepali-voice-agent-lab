"""Background sampler for CPU / RAM (and CUDA memory/utilisation when torch+CUDA are in use)."""

from __future__ import annotations

import statistics
import sys
import threading
from typing import Any

import psutil


def _cuda() -> Any:
    torch = sys.modules.get("torch")  # only if a provider already imported it; never import torch here
    return torch if torch is not None and torch.cuda.is_available() else None


class ResourceMonitor:
    """Note: for server-backed models (Ollama/vLLM) the model runs in another process, so the
    system-wide CPU/RAM columns are the meaningful ones; process columns cover only this client."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.samples: list[tuple[float, float, int, int, float | None]] = []
        self._stop = threading.Event()
        self._proc = psutil.Process()

    def _sample(self) -> None:
        util = None
        if (torch := _cuda()) is not None:
            try:
                util = float(torch.cuda.utilization())  # needs pynvml; optional
            except Exception:
                util = None
        self.samples.append(
            (
                self._proc.cpu_percent(None),
                psutil.cpu_percent(None),
                self._proc.memory_info().rss,
                psutil.virtual_memory().used,
                util,
            )
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self) -> ResourceMonitor:
        self._proc.cpu_percent(None)
        psutil.cpu_percent(None)
        if (torch := _cuda()) is not None:
            torch.cuda.reset_peak_memory_stats()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        if not self.samples:
            self._sample()

    def summary(self) -> dict[str, Any]:
        proc_cpu, sys_cpu, rss, used, util = zip(*self.samples, strict=True)
        gpu_utils = [u for u in util if u is not None]
        torch = _cuda()
        return {
            "proc_cpu_avg_pct": round(statistics.fmean(proc_cpu), 1),
            "sys_cpu_avg_pct": round(statistics.fmean(sys_cpu), 1),
            "proc_peak_rss_mb": round(max(rss) / 1e6),
            "sys_ram_peak_used_mb": round(max(used) / 1e6),
            "gpu_peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1e6) if torch is not None else None,
            "gpu_util_avg_pct": round(statistics.fmean(gpu_utils), 1) if gpu_utils else None,
        }
