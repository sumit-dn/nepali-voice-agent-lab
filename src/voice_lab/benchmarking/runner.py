"""Generic benchmark runner: models x variants x samples, with caching, resource monitoring, and storage.

Kind-specific logic is just an `evaluate(provider, sample, job, run_dir)` function (see bottom of file).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from voice_lab import config
from voice_lab.audio.preprocess import load_and_prepare
from voice_lab.benchmarking.monitor import ResourceMonitor
from voice_lab.benchmarking.store import Store
from voice_lab.config import ConfigError, ModelSpec
from voice_lab.evaluation.llm_checks import hindi_markers, run_checks
from voice_lab.evaluation.manifest import resolve_audio
from voice_lab.evaluation.metrics import char_errors, score_asr, word_errors
from voice_lab.evaluation.text import NORMALIZATION
from voice_lab.llm.tools import run_with_tools
from voice_lab.providers.base import ASRProvider, LLMProvider, Provider, ProviderError, TTSProvider
from voice_lab.providers.registry import build
from voice_lab.utils.hardware import system_info

log = logging.getLogger(__name__)

Evaluate = Callable[[Provider, dict[str, Any], "Job", Path], tuple[dict[str, Any], str, float]]


@dataclass
class Job:
    spec: ModelSpec
    variant: str = "default"  # ASR: preprocessing profile; TTS: voice
    config: dict[str, Any] = field(default_factory=dict)  # everything besides the model that changes outputs

    @property
    def cache_key(self) -> str:
        cfg = hashlib.sha256(json.dumps(self.config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]
        return f"{self.spec.fingerprint()}-{cfg}"


def _pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 4) if values else None


def summarize(kind: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [s for s in samples if not s.get("error")]
    lat = [s["latency"] for s in ok if s.get("latency") is not None]
    out: dict[str, Any] = {
        "n": len(samples),
        "failed": len(samples) - len(ok),
        "latency_mean": round(float(np.mean(lat)), 4) if lat else None,
        "latency_p50": _pct(lat, 50),
        "latency_p95": _pct(lat, 95),
    }
    m = [s["metrics"] for s in ok]
    if kind == "asr" and m:
        words = sum(x["ref_words"] for x in m)
        chars = sum(x["ref_chars"] for x in m)
        out["wer"] = round(sum(x["sub"] + x["del"] + x["ins"] for x in m) / words, 4) if words else None
        out["cer"] = round(sum(x["char_errors"] for x in m) / chars, 4) if chars else None
        out["rtf_mean"] = round(float(np.mean([x["rtf"] for x in m])), 4)
        hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for x in m:
            for kind_, (h, t) in x["entities"].items():
                hits[kind_][0] += h
                hits[kind_][1] += t
        out["entity_accuracy"] = {
            k: {"acc": round(h / t, 4) if t else None, "hits": h, "total": t} for k, (h, t) in hits.items()
        }
        out["error_tags"] = dict(Counter(tag for x in m for tag in x["tags"]).most_common())
    elif kind == "llm" and m:
        out["checks_passed"] = sum(x["passed"] for x in m)
        out["checks_total"] = sum(x["total"] for x in m)
        out["pass_rate"] = round(out["checks_passed"] / out["checks_total"], 4) if out["checks_total"] else None
        per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for x in m:
            for name, passed in x["checks"].items():
                key = name.split(":")[0]
                per[key][0] += int(passed)
                per[key][1] += 1
        out["check_pass_rates"] = {k: round(p / t, 4) for k, (p, t) in per.items()}
        ttft = [x["ttft"] for x in m if x.get("ttft") is not None]
        out["ttft_p50"], out["ttft_p95"] = _pct(ttft, 50), _pct(ttft, 95)
        tps = [x["tokens_per_second"] for x in m if x.get("tokens_per_second")]
        out["tokens_per_second_mean"] = round(float(np.mean(tps)), 2) if tps else None
        out["human_eval_pending"] = sum(bool(x.get("human_eval")) for x in m)
        # computed from stored reply text, so cached results get it too
        leaks = [s["id"] for s in ok if hindi_markers(s.get("output") or "")]
        out["hindi_leak_rate"] = round(len(leaks) / len(ok), 4)
        out["hindi_leak_samples"] = leaks
    elif kind == "tts" and m:
        out["rtf_mean"] = round(float(np.mean([x["rtf"] for x in m])), 4)
        out["audio_seconds_total"] = round(sum(x["audio_seconds"] for x in m), 2)
        out["sample_rate"] = m[0]["sample_rate"]
        judged = [x["asr_judge"] for x in m if x.get("asr_judge")]
        if judged:
            chars = sum(j["ref_chars"] for j in judged)
            out["asr_judge_cer"] = round(sum(j["char_errors"] for j in judged) / chars, 4) if chars else None
            out["asr_judge_model"] = judged[0]["model"]
    return out


def _flatten(prefix: str, value: Any, row: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}_{k}" if prefix else k, v, row)
    else:
        row[prefix] = value


def run_benchmark(
    kind: str,
    rows: list[dict[str, Any]],
    jobs: list[Job],
    evaluate: Evaluate,
    *,
    dataset: Path,
    dataset_sha: str,
    output: Path | None = None,
    workers: int = 1,
    force: bool = False,
    device: str | None = None,
    allow_cloud: bool = False,
    run_config: dict[str, Any] | None = None,
) -> Path:
    started = datetime.now(UTC)
    run_id = f"{started:%Y%m%dT%H%M%S}{started.microsecond // 1000:03d}Z-{kind}"  # ms: back-to-back runs stay unique
    run_dir = (output or config.RESULTS_DIR / kind) / run_id
    run_dir.mkdir(parents=True)
    store = Store(config.RESULTS_DIR / "lab.db")
    hardware = system_info()
    run_config = {**(run_config or {}), "workers": workers, "device": device, "force": force}
    store.add_run(run_id, kind, str(dataset), dataset_sha, run_dir, run_config, hardware)
    summary_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    current: Provider | None = None  # reused across variants of the same model (e.g. telephone profiles)
    for job in jobs:
        spec, key = job.spec, job.cache_key
        results: dict[str, dict[str, Any]] = {}
        todo = []
        for row in rows:
            hit = None if force else store.cached(key, dataset_sha, row["id"])
            if hit:
                results[row["id"]] = {
                    "id": row["id"],
                    "category": row.get("category"),
                    "latency": hit["latency"],
                    "metrics": json.loads(hit["metrics"]),
                    "output": hit["output"],
                    "error": None,
                    "reused_from": hit["reused_from"] or hit["run_id"],
                }
            else:
                todo.append(row)
        resources: dict[str, Any] = {}
        load_seconds = None
        error = None
        label = f"{spec.id} [{job.variant}]"
        if todo:
            try:
                if current is None or current.spec is not spec:
                    current = None  # drop the previous model before loading the next one
                    current = build(kind, spec.id, spec=spec, device=device, allow_cloud=allow_cloud)
                provider = current
                with ResourceMonitor() as monitor:
                    provider.load()

                    def one(
                        row: dict[str, Any], provider: Provider = provider, job: Job = job, label: str = label
                    ) -> dict[str, Any]:
                        base = {"id": row["id"], "category": row.get("category"), "reused_from": None}
                        try:
                            metrics, output_text, latency = evaluate(provider, row, job, run_dir)
                            return {
                                **base,
                                "latency": latency,
                                "metrics": metrics,
                                "output": output_text,
                                "error": None,
                            }
                        except Exception as e:  # one bad sample must not kill the run; it is recorded
                            log.warning("%s sample %s failed: %s", label, row["id"], e)
                            return {
                                **base,
                                "latency": None,
                                "metrics": None,
                                "output": None,
                                "error": f"{type(e).__name__}: {e}",
                            }

                    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                        for result in pool.map(one, todo):
                            results[result["id"]] = result
                resources = monitor.summary()
                load_seconds = provider.load_seconds
            except (ProviderError, ConfigError, ImportError, OSError) as e:
                error = str(e)
                log.error("%s: %s", label, e)
        samples = [results[r["id"]] for r in rows if r["id"] in results]
        for s in samples:
            if not s.get("reused_from"):
                store.add_sample(run_id, spec.id, job.variant, key, dataset_sha, s)
        summary = summarize(kind, samples)
        record = {
            "run_id": run_id,
            "kind": kind,
            "model": asdict(spec),
            "variant": job.variant,
            "job_config": job.config,
            "cache_key": key,
            "dataset": str(dataset),
            "dataset_sha": dataset_sha,
            "hardware": hardware,
            "run_config": run_config,
            "load_seconds": load_seconds,
            "resources": resources,
            "error": error,
            "summary": summary,
            "samples": samples,
        }
        safe_variant = job.variant.replace("/", "_")
        (run_dir / f"{spec.id}__{safe_variant}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        store.add_model(run_id, record)
        reused = sum(bool(s.get("reused_from")) for s in samples)
        print(f"  {label}: {'ERROR ' + error if error else f'{len(samples)} samples ({reused} reused from cache)'}")

        for category in [None, *sorted({s.get("category") or "" for s in samples})]:
            subset = samples if category is None else [s for s in samples if (s.get("category") or "") == category]
            row_out: dict[str, Any] = {"model": spec.id, "variant": job.variant, "category": category or "ALL"}
            _flatten("", summarize(kind, subset), row_out)
            row_out.update(resources if category is None else {})
            row_out["load_seconds"] = load_seconds if category is None else None
            row_out["license"] = spec.license.get("id")
            row_out["commercial_use"] = spec.commercial_use
            summary_rows.append(row_out)
        for s in samples:
            sample_rows.append(
                {
                    "model": spec.id,
                    "variant": job.variant,
                    "id": s["id"],
                    "category": s.get("category"),
                    "latency": s.get("latency"),
                    "output": s.get("output"),
                    "error": s.get("error"),
                    "reused_from": s.get("reused_from"),
                }
            )

    for name, table in (("summary.csv", summary_rows), ("samples.csv", sample_rows)):
        if table:
            fields = list(dict.fromkeys(k for r in table for k in r))
            with (run_dir / name).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(table)
    return run_dir


# --- kind-specific evaluation -------------------------------------------------------------------


def eval_asr(provider: Provider, row: dict[str, Any], job: Job, run_dir: Path) -> tuple[dict[str, Any], str, float]:
    assert isinstance(provider, ASRProvider)
    audio = load_and_prepare(resolve_audio(row["audio"]), job.config["steps"])
    result = provider.transcribe(audio)
    metrics = score_asr(row, result.text)
    metrics.update(
        rtf=result.real_time_factor,
        audio_seconds=result.audio_duration_seconds,
        input_conversion=result.metadata.get("model_input_conversion"),
    )
    return metrics, result.text, result.latency_seconds


def llm_messages(row: dict[str, Any], llm_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    system = llm_cfg.get("system_prompt", "")
    if kb := row.get("context_file"):
        system += "\n\n# Knowledge base\n" + resolve_audio(kb).read_text(encoding="utf-8")
    messages = [{"role": "system", "content": system}] if system else []
    return messages + (row.get("messages") or [{"role": "user", "content": row["prompt"]}])


def eval_llm(provider: Provider, row: dict[str, Any], job: Job, run_dir: Path) -> tuple[dict[str, Any], str, float]:
    assert isinstance(provider, LLMProvider)
    messages = llm_messages(row, job.config)
    if row.get("tools"):
        result, trace, ttft, total = run_with_tools(provider, messages)
    else:
        result = provider.generate(messages)
        trace, ttft, total = [], result.time_to_first_token, result.latency_seconds
    checks = run_checks(row.get("expect", {}), result.text, trace)
    out_tokens = result.output_tokens
    gen_time = result.latency_seconds - (result.time_to_first_token or 0)
    metrics = {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "ttft": ttft,
        "input_tokens": result.input_tokens,
        "output_tokens": out_tokens,
        "tokens_per_second": round(out_tokens / gen_time, 2) if out_tokens and gen_time > 0 else None,
        "tool_calls": trace,
        "human_eval": row.get("human_eval"),
        "llm_metadata": result.metadata,
    }
    return metrics, result.text, total


def make_eval_tts(judge: ASRProvider | None) -> Evaluate:
    def eval_tts(provider: Provider, row: dict[str, Any], job: Job, run_dir: Path) -> tuple[dict[str, Any], str, float]:
        assert isinstance(provider, TTSProvider)
        voice = None if job.variant == "default" else job.variant
        out = run_dir / "audio" / provider.spec.id / job.variant.replace("/", "_") / row["id"]
        result = provider.synthesize(row["text"], out, voice=voice)
        metrics: dict[str, Any] = {
            "rtf": result.real_time_factor,
            "audio_seconds": result.audio_duration_seconds,
            "sample_rate": result.sample_rate,
            "text": row["text"],
            "audio_path": result.audio_path,
            "original_path": result.original_path,
        }
        if (
            judge is not None
        ):  # intelligibility proxy: can an ASR model recover the input text? (ASR errors confound it)
            from voice_lab.audio.io import load_audio

            heard = judge.transcribe(load_audio(result.audio_path))
            words, _ = word_errors(row["text"], heard.text)
            metrics["asr_judge"] = {
                "model": judge.spec.id,
                "text": heard.text,
                **char_errors(row["text"], heard.text),
                **words,
            }
        return metrics, result.audio_path, result.latency_seconds

    return eval_tts


def asr_jobs(specs: list[ModelSpec], profiles: dict[str, list[dict[str, Any]]]) -> list[Job]:
    return [
        Job(s, name, {"steps": steps, "normalization": NORMALIZATION})
        for s in specs
        for name, steps in profiles.items()
    ]
