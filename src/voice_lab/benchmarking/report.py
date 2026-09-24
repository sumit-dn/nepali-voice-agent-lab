"""Markdown reports from the latest run of each kind + the registries. Evidence tables only - no winner."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

from voice_lab import config
from voice_lab.config import KINDS, load_models
from voice_lab.utils.hardware import PACKAGES, system_info

DISCLAIMER = (
    "> Evidence only. This report does not rank models or pick a winner. Weigh quality, latency, "
    "hardware, concurrency and licensing for the actual use case. Numbers come only from runs listed below."
)
CORE_PACKAGES = ("numpy", "scipy", "soundfile", "PyYAML", "psutil", *PACKAGES)


def latest_run(kind: str, base: Path | None = None) -> Path | None:
    root = (base or config.RESULTS_DIR) / kind
    runs = sorted(p for p in root.glob("*T*Z-*") if p.is_dir()) if root.exists() else []
    return runs[-1] if runs else None


def load_records(run_dir: Path | None) -> list[dict[str, Any]]:
    """Run records with summaries recomputed from stored samples, so metric fixes apply to old runs too."""
    from voice_lab.benchmarking.runner import summarize

    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("*.json"))] if run_dir else []
    for r in records:
        r["summary"] = summarize(r["kind"], r["samples"])
    return records


def fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _header(title: str, run_dir: Path | None, records: list[dict[str, Any]]) -> list[str]:
    out = [f"# {title}", "", DISCLAIMER, ""]
    if not records:
        return [*out, "_No results yet. Run the matching `voice-lab benchmark` command first._"]
    hw = records[0]["hardware"]
    gpu = ", ".join(g["name"] for g in hw.get("gpus", [])) or "none (CPU-only)"
    out += [
        f"- Run: `{run_dir}`",
        f"- Dataset: `{records[0]['dataset']}` (sha {records[0]['dataset_sha']})",
        f"- Hardware: {hw['cpu']} ({hw['cpu_threads']} threads), {hw['ram_total_gb']} GB RAM, GPU: {gpu}",
        "",
    ]
    return out


def asr_report(run_dir: Path | None) -> str:
    records = load_records(run_dir)
    out = _header("ASR comparison", run_dir, records)
    if not records:
        return "\n".join(out)
    rows = []
    for r in records:
        s, res = r["summary"], r["resources"] or {}
        rows.append(
            [
                r["model"]["id"],
                r["variant"],
                s["n"],
                s["failed"],
                s.get("wer"),
                s.get("cer"),
                s.get("latency_p50"),
                s.get("latency_p95"),
                s.get("rtf_mean"),
                res.get("proc_peak_rss_mb"),
                res.get("gpu_peak_mem_mb"),
                r["load_seconds"],
                r["model"]["license"].get("id"),
                r["model"]["license"].get("commercial_use"),
                r["error"],
            ]
        )
    out += [
        "## Overall",
        "",
        table(
            [
                "Model",
                "Preprocessing",
                "N",
                "Failed",
                "WER",
                "CER",
                "Latency p50 (s)",
                "p95 (s)",
                "RTF",
                "Peak RAM (MB)",
                "Peak VRAM (MB)",
                "Load (s)",
                "License",
                "Commercial",
                "Error",
            ],
            rows,
        ),
        "",
    ]
    out += ["## WER by category", ""]
    by_cat: dict[tuple[str, str], dict[str, Any]] = {}
    for r in records:
        for s in r["samples"]:
            by_cat.setdefault((r["model"]["id"], r["variant"]), {}).setdefault(s.get("category") or "-", []).append(s)
    from voice_lab.benchmarking.runner import summarize

    cats = sorted({c for d in by_cat.values() for c in d})
    out += [
        table(
            ["Model", "Preprocessing", *cats],
            [
                [m, v, *[summarize("asr", d[c]).get("wer") if c in d else None for c in cats]]
                for (m, v), d in by_cat.items()
            ],
        ),
        "",
    ]
    out += [
        "WER/CER are corpus-level over normalised text (" + records[0]["job_config"].get("normalization", "") + ")."
    ]
    return "\n".join(out)


def asr_error_report(run_dir: Path | None, examples: int = 3) -> str:
    records = load_records(run_dir)
    out = _header("ASR error analysis", run_dir, records)
    for r in records:
        tags = r["summary"].get("error_tags") or {}
        out += [
            f"## {r['model']['id']} [{r['variant']}]",
            "",
            "Error tag counts (samples containing each error type): "
            + (", ".join(f"{k}={v}" for k, v in tags.items()) or "none"),
            "",
        ]
        for tag in tags:
            hits = [s for s in r["samples"] if s.get("metrics") and tag in s["metrics"]["tags"]][:examples]
            out.append(f"### {tag}")
            for s in hits:
                pairs = "; ".join(f"{a or '∅'}→{b or '∅'}" for a, b in s["metrics"]["error_pairs"][:8])
                out += [
                    "",
                    f"- **{s['id']}** ({s.get('category')})",
                    f"  - Prediction: {s['output']}",
                    f"  - Word errors: {pairs}",
                ]
            out.append("")
    return "\n".join(out)


def entity_report(run_dir: Path | None) -> str:
    records = load_records(run_dir)
    out = _header("Critical-entity accuracy (numbers, names, dates, ...)", run_dir, records)
    if not records:
        return "\n".join(out)
    kinds = sorted({k for r in records for k in (r["summary"].get("entity_accuracy") or {})})
    rows = []
    for r in records:
        acc = r["summary"].get("entity_accuracy") or {}
        rows.append(
            [
                r["model"]["id"],
                r["variant"],
                r["summary"].get("wer"),
                *[
                    f"{acc[k]['acc']:.2f} ({acc[k]['hits']}/{acc[k]['total']})"
                    if k in acc and acc[k]["total"]
                    else None
                    for k in kinds
                ],
            ]
        )
    out += [
        table(["Model", "Preprocessing", "WER", *kinds], rows),
        "",
        "An entity counts as correct only if it appears in the transcript after normalisation; numeric entities are "
        "compared as digit strings (Devanagari/ASCII digits and grouping ignored). Numbers spoken as words and "
        "transcribed as words will count as misses - see docs/evaluation.md.",
    ]
    return "\n".join(out)


def llm_report(run_dir: Path | None) -> str:
    records = load_records(run_dir)
    out = _header("LLM comparison", run_dir, records)
    if not records:
        return "\n".join(out)
    checks = sorted({k for r in records for k in (r["summary"].get("check_pass_rates") or {})})
    rows = [
        [
            r["model"]["id"],
            r["model"]["checkpoint"],
            r["summary"]["n"],
            r["summary"]["failed"],
            r["summary"].get("pass_rate"),
            *[(r["summary"].get("check_pass_rates") or {}).get(c) for c in checks],
            r["summary"].get("hindi_leak_rate"),
            r["summary"].get("ttft_p50"),
            r["summary"].get("latency_p50"),
            r["summary"].get("tokens_per_second_mean"),
            r["model"]["license"].get("id"),
            r["model"]["license"].get("commercial_use"),
            r["error"],
        ]
        for r in records
    ]
    out += [
        "## Deterministic checks",
        "",
        table(
            [
                "Model",
                "Checkpoint",
                "N",
                "Failed",
                "Pass rate",
                *checks,
                "Hindi-leak (heuristic)",
                "TTFT p50 (s)",
                "Latency p50 (s)",
                "Tok/s",
                "License",
                "Commercial",
                "Error",
            ],
            rows,
        ),
        "",
    ]
    out += [
        "## Needs human evaluation",
        "",
        "Automatic checks cannot judge fluency/naturalness. Rate these replies 1-5.",
        "",
    ]
    for r in records:
        for s in r["samples"]:
            if s.get("metrics") and s["metrics"].get("human_eval"):
                out.append(
                    f"- **{r['model']['id']} / {s['id']}** ({', '.join(s['metrics']['human_eval'])}): "
                    f"{fmt(s['output'])}"
                )
    return "\n".join(out)


def tts_report(run_dir: Path | None) -> str:
    records = load_records(run_dir)
    out = _header("TTS comparison", run_dir, records)
    if not records:
        return "\n".join(out)
    rows = [
        [
            r["model"]["id"],
            r["variant"],
            r["summary"]["n"],
            r["summary"]["failed"],
            r["summary"].get("latency_p50"),
            r["summary"].get("rtf_mean"),
            r["summary"].get("sample_rate"),
            r["summary"].get("asr_judge_cer"),
            (r["resources"] or {}).get("proc_peak_rss_mb"),
            r["load_seconds"],
            r["model"]["license"].get("id"),
            r["model"]["license"].get("commercial_use"),
            r["error"],
        ]
        for r in records
    ]
    out += [
        table(
            [
                "Model",
                "Voice",
                "N",
                "Failed",
                "Latency p50 (s)",
                "RTF",
                "Sample rate",
                "ASR-judge CER",
                "Peak RAM (MB)",
                "Load (s)",
                "License",
                "Commercial",
                "Error",
            ],
            rows,
        ),
        "",
        "ASR-judge CER is an intelligibility *proxy* (an ASR model transcribes the TTS output); "
        "ASR errors confound it. "
        "It is not MOS. Naturalness/pronunciation/prosody need the human rating protocol in docs/evaluation.md.",
    ]
    human = run_dir / "human_eval" / "summary.json" if run_dir else None
    if human and human.exists():
        summary = json.loads(human.read_text(encoding="utf-8"))
        from voice_lab.evaluation.human import CRITERIA

        rows = [
            [system, *[f"{v[c]['mean']} ± {v[c]['ci95']} (n={v[c]['n']})" if c in v else None for c in CRITERIA]]
            for system, v in summary["results"].items()
        ]
        out += [
            "",
            f"## Human ratings (1-5, mean ± 95% CI, {summary['raters']} rater(s))",
            "",
            table(["System", *CRITERIA], rows),
        ]
    return "\n".join(out)


def licensing_report() -> str:
    out = [
        "# Licensing audit",
        "",
        "Generated from `configs/*.yaml`. Every value was copied from the checkpoint's own card/licence on the "
        "`verified` date. **REVIEW REQUIRED** means we could not establish commercial rights - ask legal.",
        "",
        "Terms are not interchangeable:",
        "",
        "- **Open-source**: an OSI-approved licence (Apache-2.0, MIT, ...) on the weights *and* code.",
        "- **Open-weight**: weights downloadable under a custom licence (Llama, Gemma ToU) with use restrictions.",
        "- **Free**: no fee. Says nothing about commercial rights.",
        "- **Commercially usable**: the exact licence of *this checkpoint* (and its training data, where it "
        "propagates) permits our commercial use.",
        "",
    ]
    for kind in KINDS:
        rows = []
        for m in load_models(kind).values():
            lic = m.license
            rows.append(
                [
                    m.id,
                    m.checkpoint,
                    m.revision,
                    lic.get("id"),
                    lic.get("commercial_use", "REVIEW REQUIRED"),
                    lic.get("url"),
                    lic.get("notes"),
                    m.verified,
                ]
            )
        out += [
            f"## {kind.upper()}",
            "",
            table(
                [
                    "ID",
                    "Checkpoint",
                    "Revision",
                    "Licence",
                    "Commercial use",
                    "Reference",
                    "Restrictions / notes",
                    "Verified",
                ],
                rows,
            ),
            "",
        ]
    rows = []
    for name in CORE_PACKAGES:
        try:
            md: Any = importlib.metadata.metadata(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        classifiers = [c.split(" :: ")[-1] for c in md.get_all("Classifier") or [] if c.startswith("License ::")]
        first_line = (md.get("License") or "").split("\n")[0][:60]
        pkg_license = md.get("License-Expression") or ", ".join(classifiers) or first_line  # License may be full text
        rows.append([name, md.get("Version"), pkg_license or "REVIEW REQUIRED"])
    out += [
        "## Runtime dependencies (from installed package metadata)",
        "",
        table(["Package", "Version", "Licence"], rows),
        "",
        "piper-tts is GPL-3.0-or-later: fine for internal research; shipping it inside a product needs legal review.",
    ]
    return "\n".join(out)


def hardware_report(runs: dict[str, Path | None]) -> str:
    hw = system_info()
    out = [
        "# Hardware profile",
        "",
        "## This machine",
        "",
        "```",
        json.dumps(hw, indent=2),
        "```",
        "",
        "## Per-model load time and peak resources (latest run of each kind)",
        "",
    ]
    rows = []
    for kind, run_dir in runs.items():
        for r in load_records(run_dir):
            res = r["resources"] or {}
            rows.append(
                [
                    kind,
                    r["model"]["id"],
                    r["variant"],
                    r["load_seconds"],
                    res.get("proc_peak_rss_mb"),
                    res.get("sys_ram_peak_used_mb"),
                    res.get("proc_cpu_avg_pct"),
                    res.get("gpu_peak_mem_mb"),
                    res.get("gpu_util_avg_pct"),
                    r["model"].get("hardware"),
                ]
            )
    out += [
        table(
            [
                "Kind",
                "Model",
                "Variant",
                "Load (s)",
                "Peak proc RAM (MB)",
                "Peak sys RAM used (MB)",
                "Proc CPU avg %",
                "Peak VRAM (MB)",
                "GPU util %",
                "Documented requirement",
            ],
            rows,
        ),
        "",
        "For Ollama/vLLM models the weights live in the server process: use the system RAM column and the "
        "server's own reporting (`ollama ps`).",
    ]
    return "\n".join(out)


def generate(out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or config.REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {k: latest_run(k) for k in ("asr", "llm", "tts")}
    telephone = latest_run("telephone")
    docs = {
        "asr-comparison.md": asr_report(runs["asr"]),
        "asr-error-analysis.md": asr_error_report(runs["asr"]),
        "critical-entity-accuracy.md": entity_report(runs["asr"]),
        "telephone-comparison.md": asr_report(telephone).replace(
            "# ASR comparison", "# Telephone-audio ASR comparison", 1
        ),
        "llm-comparison.md": llm_report(runs["llm"]),
        "tts-comparison.md": tts_report(runs["tts"]),
        "licensing.md": licensing_report(),
        "hardware-profile.md": hardware_report(runs),
    }
    paths = []
    for name, text in docs.items():
        path = out_dir / name
        path.write_text(text + "\n", encoding="utf-8")
        paths.append(path)
    return paths
