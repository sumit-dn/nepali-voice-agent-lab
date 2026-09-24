"""Local web UI (Gradio) over the same providers the CLI uses. Start with `voice-lab ui`.

Local by construction: binds to 127.0.0.1, no share link, Gradio analytics off, system fonts only (no Google Fonts),
weights load only from the local cache, and cloud models still need the "Allow cloud models" switch.
Benchmarks and downloads run the real `voice-lab` CLI in a subprocess, so their logic is not duplicated here.
Layout: sidebar (runtime, loaded models, system) + Playground / Evaluate / Models. Styles live in ui.css.
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # must precede `import gradio`
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never fetch weights implicitly (must precede HF imports)

import contextlib
import csv
import gc
import html
import json
import queue
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

import gradio as gr

from voice_lab import config
from voice_lab.audio.io import load_audio, peak_dbfs
from voice_lab.audio.preprocess import load_and_prepare
from voice_lab.config import KINDS, ConfigError, load_benchmark_config, load_models
from voice_lab.download import local_status, refresh_status
from voice_lab.evaluation.human import CRITERIA, import_ratings, make_sheet, sample_texts
from voice_lab.evaluation.manifest import ManifestError, load_manifest
from voice_lab.llm.tools import run_with_tools
from voice_lab.providers.base import Provider, ProviderError
from voice_lab.providers.registry import build

READY = {"downloaded", "pulled", "bundled with pip pkg"}
SILENCE_DBFS = -60.0  # quieter than this is treated as "the mic recorded nothing"
USER_ERRORS = (ConfigError, ProviderError, ManifestError, ValueError, FileNotFoundError)
NOISE = ("FRAME_DURATION_MS", "Missing phoneme", "warnings.warn", "Warning:", "Fetching ", "Next: voice-lab reports")
NO_VAD = ("None (skip VAD)", "")
HEAVY: dict[str, Any] = {"concurrency_limit": 1, "concurrency_id": "models"}  # one model job at a time, app-wide
SCALE = "1 = bad · 2 = poor · 3 = acceptable · 4 = good · 5 = excellent"
CRITERIA_HELP = {
    "naturalness": "Sounds like a real person speaking, not a machine.",
    "pronunciation": "Nepali words, names and numbers are pronounced correctly.",
    "clarity": "Clean audio: no noise, clicks, glitches or muffled words.",
    "prosody": "Rhythm, stress and intonation fit the sentence (questions rise, pauses fall in place).",
    "accent": "Sounds like a native Nepali speaker, not a Hindi or foreign accent.",
    "intelligibility": "Every word can be understood on the first listen.",
    "mixed_language": "English words inside Nepali sound natural. Leave empty if the clip has no English.",
}
CSS = Path(__file__).with_name("ui.css")
STATUS_TONE = {
    "downloaded": "ok",
    "pulled": "ok",
    "bundled with pip pkg": "ok",
    "cloud (opt-in)": "warn",
    "ollama unreachable": "bad",
}
COMMERCIAL_TONE = {"yes": "ok", "no": "bad", "conditional": "warn", "REVIEW REQUIRED": "warn"}
FONT = ["Inter", "Segoe UI", "system-ui", "-apple-system", "Noto Sans", "Noto Sans Devanagari", "Ubuntu", "sans-serif"]
MONO = ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"]
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=FONT,
    font_mono=MONO,
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#f8fafc",
    body_background_fill_dark="#0b1220",
    block_background_fill="#ffffff",
    block_background_fill_dark="#0f172a",
    block_border_color="#e2e8f0",
    block_border_color_dark="#1e293b",
    block_shadow="none",
    block_label_text_weight="600",
    input_background_fill_dark="#0b1220",
    button_primary_background_fill="#1e40af",
    button_primary_background_fill_hover="#1e3a8a",
    button_primary_background_fill_dark="#2563eb",
    button_primary_background_fill_hover_dark="#1d4ed8",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
)
CHAT_EXAMPLES = [
    "शनिबार office खुल्छ?",
    "मेरो account balance check गरिदिनुहोस्। Customer ID C-1001।",
    "Please मेरो appointment A-501 cancel गरिदिनुहोस्।",
    "What are your office hours?",
]

# ponytail: one global model cache + queue concurrency 1 (one heavy job at a time); fine for a single local
# user on a 16 GB box, needs per-session limits before anyone shares this UI.
_providers: dict[tuple[Any, ...], Provider] = {}
_lock = threading.Lock()


# --- Small HTML helpers ----------------------------------------------------------------------------


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def esc(value: Any) -> str:
    return html.escape(str(value))


def pill(text: str, tone: str = "") -> str:
    return f'<span class="vl-pill {tone}">{esc(text)}</span>'


def metrics_html(items: list[tuple[str, str]], note: str = "") -> str:
    cells = "".join(
        f'<div class="vl-metric"><span class="k">{esc(k)}</span><span class="v">{esc(v)}</span></div>' for k, v in items
    )
    return f'<div class="vl-metrics">{cells}</div>' + (f'<p class="vl-note">{esc(note)}</p>' if note else "")


def empty_html(text: str) -> str:
    return f'<div class="vl-empty">{esc(text)}</div>'


def section(title: str, hint: str = "") -> None:
    gr.HTML(
        f'<div class="vl-section">{esc(title)}</div>' + (f'<p class="vl-hint">{esc(hint)}</p>' if hint else ""),
        elem_classes=["vl-tight"],
    )


def friendly(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Show expected failures (not downloaded, server down, bad input) as a UI message, not a traceback."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except USER_ERRORS as e:
            raise gr.Error(str(e)) from e

    return wrapper


# --- Models in memory ------------------------------------------------------------------------------


def provider(kind: str, model_id: str, device: str, allow_cloud: bool) -> Any:
    if not model_id:
        raise gr.Error(f"Choose a {kind.upper()} model first.")
    key = (kind, model_id, device, allow_cloud)
    with _lock:
        if key not in _providers:
            p = build(kind, model_id, device=None if device == "auto" else device, allow_cloud=allow_cloud)
            p.load()
            _providers[key] = p
        return _providers[key]


def unload() -> str:
    """Drop cached models; Ollama keeps weights in its own process, so ask it to free them too."""
    from voice_lab.llm.openai_compat import OpenAICompatLLM
    from voice_lab.utils.http import open_url

    with _lock:
        cached = list(_providers.values())
        _providers.clear()
    for p in cached:
        if isinstance(p, OpenAICompatLLM) and p.spec.params.get("server") == "ollama":
            body = {"model": p.spec.checkpoint, "keep_alive": 0}
            with contextlib.suppress(ProviderError), open_url(f"{p.root}/api/generate", body=body, timeout=10):
                pass  # server already down = nothing loaded there
    gc.collect()
    return loaded_html()


def loaded_html() -> str:
    with _lock:
        keys = list(_providers)
    if not keys:
        return '<p class="vl-hint">No models in memory.</p>'
    rows = "".join(f"<div>{pill(kind.upper(), 'info')} {esc(model_id)}</div>" for kind, model_id, *_ in keys)
    return f'<div class="vl-loaded">{rows}</div>'


def choices(kind: str) -> list[tuple[str, str]]:
    return [(f"{m.id}  ·  {local_status(m)}", m.id) for m in load_models(kind).values() if m.provider != "none"]


def default(kind: str, preferred: str | None = None) -> str | None:
    models = load_models(kind)
    if preferred in models and local_status(models[preferred]) in READY:
        return preferred
    return next((m.id for m in models.values() if m.enabled and local_status(m) in READY), None)


def out_wav(tag: str) -> Path:
    return config.RESULTS_DIR / "ui" / f"{datetime.now(UTC):%Y%m%dT%H%M%S%f}-{tag}.wav"


def require_sound(path: str | None) -> str:
    if not path:
        raise gr.Error("Record or upload audio first.")
    level = peak_dbfs(load_audio(path))
    if level < SILENCE_DBFS:
        raise gr.Error(
            f"The recording is silent (peak {level:.0f} dBFS), nothing to recognise. Check that a microphone "
            "is plugged in, selected in the browser and not muted (Linux: `alsamixer` → F4 Capture), then record again."
        )
    return path


def system_prompt() -> str:
    return str(load_benchmark_config().get("llm", {}).get("system_prompt", ""))


# --- System & models -------------------------------------------------------------------------------


def system_state() -> dict[str, Any]:
    from voice_lab.config import OPTIONAL_ENV, missing_credentials
    from voice_lab.utils.hardware import system_info
    from voice_lab.utils.http import get_json

    info = system_info()
    root = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        info["ollama"] = get_json(f"{root}/api/version", timeout=3).get("version")
    except ProviderError:
        info["ollama"] = None
    missing = missing_credentials()
    info["keys"] = {k: k not in missing for k in OPTIONAL_ENV}
    info["ready"] = sum(local_status(m) in READY for kind in KINDS for m in load_models(kind).values())
    return info


def status_pills(info: dict[str, Any]) -> str:
    gpu = info["gpus"][0]["name"] if info["gpus"] else None
    return (
        '<div class="vl-pills">'
        + "".join(
            [
                pill("Local only · 127.0.0.1", "info"),
                pill(f"GPU: {gpu}", "ok") if gpu else pill("CPU only", "warn"),
                pill(f"Ollama {info['ollama']}", "ok") if info["ollama"] else pill("Ollama offline", "bad"),
                pill(f"{info['ready']} models ready", "ok" if info["ready"] else "warn"),
            ]
        )
        + "</div>"
    )


def sidebar_html(info: dict[str, Any]) -> str:
    gpu = ", ".join(f"{g['name']} ({g['vram_gb']} GB)" for g in info["gpus"]) or "none"
    keys = ", ".join(k for k in ("GEMINI_API_KEY", "HF_TOKEN") if info["keys"].get(k)) or "none"
    rows = [
        ("CPU", f"{info['cpu_threads']} threads"),
        ("RAM free", f"{info['ram_available_gb']} / {info['ram_total_gb']} GB"),
        ("Disk free", f"{info['disk_free_gb']} GB"),
        ("GPU", gpu),
        ("Ollama", info["ollama"] or "offline"),
        ("Keys set", keys),
    ]
    return '<dl class="vl-kv">' + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows) + "</dl>"


def models_rows(kind: str = "all", only_ready: bool = False) -> list[list[Any]]:
    rows = []
    for k in KINDS:
        if kind not in ("all", k):
            continue
        for m in load_models(k).values():
            status = local_status(m)
            if only_ready and status not in READY:
                continue
            rows.append(
                [
                    k.upper(),
                    m.id,
                    m.provider,
                    pill(status, STATUS_TONE.get(status, "")),
                    pill(m.commercial_use, COMMERCIAL_TONE.get(m.commercial_use, "")),
                    m.size or "",
                    "on" if m.enabled else "off",
                ]
            )
    return rows


def downloadable() -> list[str]:
    return [
        m.id for k in KINDS for m in load_models(k).values() if m.provider not in ("none", "silero") and not m.external
    ]


def run_cli(args: list[str], offline: bool = True, stdin: str | None = None) -> Iterator[str]:
    """Stream a `voice-lab` subprocess's output (known noisy warnings filtered)."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if not offline:
        env.pop("HF_HUB_OFFLINE", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "voice_lab", *args],
        cwd=config.ROOT,
        env=env,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(stdin or "")
    proc.stdin.close()
    log = f"$ voice-lab {' '.join(args)}\n"
    try:
        yield log
        for line in proc.stdout:
            if not any(n in line for n in NOISE):
                log += line
                yield log
        proc.wait()
        yield log + f"\n[finished, exit code {proc.returncode}]"
    finally:  # Stop / closed tab: Gradio closes this generator (GeneratorExit at a yield)
        if proc.poll() is None:
            proc.terminate()


def check_download(model_id: str) -> Iterator[str]:
    if not model_id:
        raise gr.Error("Choose a model first.")
    yield from run_cli(["download", model_id], offline=False, stdin="n\n")  # prints size/licence, then cancels


def do_download(model_id: str, confirmed: bool) -> Iterator[str]:
    if not model_id:
        raise gr.Error("Choose a model first.")
    if not confirmed:
        raise gr.Error("Tick the box to confirm you checked the size and licence.")
    yield from run_cli(["download", model_id, "--yes"], offline=False)


# --- Playground --------------------------------------------------------------------------------------


@friendly
def transcribe(audio: str | None, model_id: str, profile: str, device: str, allow_cloud: bool) -> tuple[str, str]:
    audio = require_sound(audio)
    steps = load_benchmark_config()["preprocess_profiles"][profile]
    p = provider("asr", model_id, device, allow_cloud)
    r = p.transcribe(load_and_prepare(audio, steps))
    conversion = r.metadata.get("model_input_conversion")
    note = f"{r.model} · profile {profile}" + (f" · model input: {', '.join(conversion)}" if conversion else "")
    return r.text, metrics_html(
        [
            ("Latency", f"{r.latency_seconds:.2f} s"),
            ("Audio", f"{r.audio_duration_seconds:.2f} s"),
            ("RTF", f"{r.real_time_factor:.2f}"),
            ("Load", f"{p.load_seconds:.1f} s"),
        ],
        note,
    )


@friendly
def detect(audio: str | None, model_id: str, device: str, allow_cloud: bool) -> tuple[list[list[float]], str]:
    audio = require_sound(audio)
    r = provider("vad", model_id, device, allow_cloud).detect_speech(load_audio(audio))
    rows = [[round(s, 2), round(e, 2), round(e - s, 2)] for s, e in r.segments]
    return rows, metrics_html(
        [
            ("Segments", str(len(rows))),
            ("Speech", f"{r.speech_seconds:.2f} s"),
            ("Audio", f"{r.audio_duration_seconds:.2f} s"),
            ("Time", f"{r.latency_seconds * 1000:.0f} ms"),
        ]
    )


@friendly
def speak(
    text: str, model_id: str, voice: str, reference: str | None, device: str, allow_cloud: bool
) -> tuple[str, str]:
    if not text.strip():
        raise gr.Error("Type some text first.")
    if reference:
        if model_id and load_models("tts").get(model_id) and load_models("tts")[model_id].provider != "xtts":
            raise gr.Error("A reference voice only works with XTTS models. Clear it, or pick an XTTS model.")
        voice = require_sound(reference)  # XTTS zero-shot cloning from an uploaded/recorded clip
    p = provider("tts", model_id, device, allow_cloud)
    r = p.synthesize(text, out_wav(model_id), voice=voice.strip() or None)
    return r.audio_path, metrics_html(
        [
            ("Latency", f"{r.latency_seconds:.2f} s"),
            ("Audio", f"{r.audio_duration_seconds:.2f} s"),
            ("RTF", f"{r.real_time_factor:.2f}"),
            ("Sample rate", f"{r.sample_rate} Hz"),
            ("Load", f"{p.load_seconds:.1f} s"),
        ],
        f"{r.model} · saved to {r.audio_path}",
    )


def _tool_messages(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": f"`{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})` → "
            f"`{json.dumps(c['result'], ensure_ascii=False)}`",
            "metadata": {"title": f"Tool call · {c['name']} (fake)"},
        }
        for c in trace
    ]


def chat(
    message: str, state: dict[str, Any] | None, model_id: str, tools: bool, system: str, device: str, allow_cloud: bool
) -> Iterator[tuple[list[dict[str, Any]], dict[str, Any], str, str]]:
    """Shows the user's message at once, then the reply. `state` keeps model-facing turns separate from the display."""
    state = state or {"turns": [], "display": []}
    if not message.strip():
        yield state["display"], state, "", ""
        return
    user = {"role": "user", "content": message}
    waiting = {
        "role": "assistant",
        "content": "Generating a reply…",
        "metadata": {"title": "Working", "status": "pending"},
    }
    yield (
        [*state["display"], user, waiting],
        state,
        "",
        metrics_html([], "Waiting for the model. On this CPU an 8B model can take 30–60 s per reply."),
    )
    try:
        llm = provider("llm", model_id, device, allow_cloud)
        turns = [*state["turns"], user]
        messages = ([{"role": "system", "content": system}] if system.strip() else []) + turns
        if tools:
            result, trace, ttft, seconds = run_with_tools(llm, messages)
        else:
            result = llm.generate(messages)
            trace, ttft, seconds = [], result.time_to_first_token, result.latency_seconds
    except (gr.Error, *USER_ERRORS) as e:
        yield state["display"], state, message, ""  # restore the chat and give the message back for a retry
        raise gr.Error(str(e)) from e
    reply = result.text or "(empty reply)"
    display = [*state["display"], user, *_tool_messages(trace), {"role": "assistant", "content": reply}]
    state = {"turns": [*turns, {"role": "assistant", "content": reply}], "display": display}
    items = [("Total", f"{seconds:.2f} s"), ("Output tokens", str(result.output_tokens or "–"))]
    if ttft is not None:
        items.insert(0, ("First token", f"{ttft:.2f} s"))
    yield display, state, "", metrics_html(items, result.model)


def clear_chat() -> tuple[list[Any], dict[str, Any], str]:
    return [], {"turns": [], "display": []}, ""


def waterfall_html(t: dict[str, Any]) -> str:
    stages = [(s, t[s]) for s in ("vad", "asr", "llm", "tts") if t.get(s) is not None]
    total = t.get("total") or sum(v for _, v in stages) or 1e-9
    rows = "".join(
        f'<span class="stage">{s.upper()}</span><div class="track"><div class="bar" '
        f'style="width:{100 * v / total:.1f}%"></div></div><span class="t">{v:.2f} s</span>'
        for s, v in stages
    )
    rows += (
        f'<span class="stage total">Total</span><span class="total"></span><span class="t total">{total:.2f} s</span>'
    )
    extras = [
        (k, f"{t[key]:.2f}{unit}")
        for k, key, unit in (
            ("ASR RTF", "asr_rtf", ""),
            ("LLM first token", "llm_ttft", " s"),
            ("TTS RTF", "tts_rtf", ""),
        )
        if t.get(key) is not None
    ]
    return f'<div class="vl-wf">{rows}</div><div style="height:10px"></div>' + metrics_html(extras)


def _stage(partial: dict[str, Any]) -> str:
    step = "Speaking…" if "response" in partial else "Generating reply…" if "transcript" in partial else "Transcribing…"
    return f'<div class="vl-pills">{pill(step, "info")}</div>'


def converse_ui(
    audio: str | None, vad_id: str, asr_id: str, llm_id: str, tts_id: str, tools: bool, device: str, allow_cloud: bool
) -> Iterator[tuple[str, str, str | None, str]]:
    """Runs the pipeline in a thread and streams each stage: transcript after ASR, reply after the LLM, audio last."""
    from voice_lab.pipeline import converse

    audio = require_sound(audio)
    updates: queue.Queue[tuple[str, Any]] = queue.Queue()

    def job() -> None:
        try:
            models = {
                "vad": provider("vad", vad_id, device, allow_cloud) if vad_id else None,
                "asr": provider("asr", asr_id, device, allow_cloud),
                "llm": provider("llm", llm_id, device, allow_cloud),
                "tts": provider("tts", tts_id, device, allow_cloud),
            }
            updates.put(("partial", {}))  # models loaded
            r = converse(
                audio,
                **models,
                out_dir=config.RESULTS_DIR / "ui" / f"conversation-{datetime.now(UTC):%Y%m%dT%H%M%S%f}",
                system_prompt=system_prompt(),
                tools=tools,
                on_update=lambda partial: updates.put(("partial", partial)),
            )
            updates.put(("done", r))
        except Exception as e:  # re-raised in the UI thread below
            updates.put(("error", e))

    threading.Thread(target=job, daemon=True).start()
    yield "", "", None, f'<div class="vl-pills">{pill("Loading models…", "info")}</div>'
    while True:
        tag, r = updates.get()
        if tag == "error":
            if isinstance(r, USER_ERRORS):
                raise gr.Error(str(r)) from r
            raise r
        if tag == "done":
            break
        yield r.get("transcript", ""), r.get("response", ""), None, _stage(r)
    details = waterfall_html(r["timings"])
    loads = ", ".join(f"{k} {v:.1f} s" for k, v in r["load_seconds"].items())
    details += f'<p class="vl-note">Model load times (not included above): {esc(loads)}</p>'
    if r.get("tool_calls"):
        details += "".join(
            f'<p class="vl-note">{esc(m["metadata"]["title"])}: {esc(m["content"])}</p>'
            for m in _tool_messages(r["tool_calls"])
        )
    if r.get("error"):
        details = f'<div class="vl-banner warn">{esc(r["error"])}</div>' + details
    yield r.get("transcript", ""), r.get("response", ""), r.get("response_audio"), details


# --- Evaluate: benchmarks, reports, human rating -----------------------------------------------------


def _manifest_kind(kind: str) -> str:
    return "asr" if kind in ("asr", "telephone") else kind


def default_dataset(kind: str) -> str:
    path = load_benchmark_config()["datasets"][_manifest_kind(kind)]
    synthetic = sorted((config.ROOT / "data" / "manifests").glob("asr_synthetic_*.jsonl"))
    if _manifest_kind(kind) == "asr" and not (config.ROOT / path).exists() and synthetic:
        return str(synthetic[0].relative_to(config.ROOT))  # no real recordings yet -> synthetic smoke set
    return str(path)


def on_benchmark_kind(kind: str) -> tuple[Any, Any, Any, Any]:
    mk = _manifest_kind(kind)
    ready = [m.id for m in load_models(mk).values() if m.enabled and local_status(m) in READY]
    return (
        gr.update(choices=choices(mk), value=ready),
        gr.update(value=default_dataset(kind)),
        gr.update(visible=mk == "asr"),
        gr.update(visible=kind == "tts", value=None),  # unset, or Gradio shows the first choice as picked
    )


def run_benchmark_ui(
    kind: str,
    models: list[str],
    dataset: str,
    limit: float | None,
    force: bool,
    profiles: list[str],
    judge: str | None,
    device: str,
    allow_cloud: bool,
) -> Iterator[tuple[str, str, Any]]:
    if not models:
        raise gr.Error("Pick at least one model.")
    args = ["benchmark", kind, *[x for m in models for x in ("--model", m)]]
    args += ["--dataset", dataset] if dataset else []
    args += ["--limit", str(int(limit))] if limit else []
    args += ["--force"] if force else []
    args += [x for p in profiles or [] for x in ("--profile", p)] if _manifest_kind(kind) == "asr" else []
    args += ["--asr-judge", judge] if kind == "tts" and judge else []
    args += ["--device", device] if device != "auto" else []
    args += ["--allow-cloud"] if allow_cloud else []
    log = ""
    for log in run_cli(args):
        yield pill("Running…", "info"), log, gr.update(visible=False)
    found = re.search(r"^Results: (.+)$", log, re.M)
    ok = found is not None and "exit code 0" in log
    table = summary_table(Path(found.group(1)) / "summary.csv") if found else None
    status = pill("Finished", "ok") if ok else pill("Failed - see the log below", "bad")
    yield status, log, gr.update(value=table, visible=table is not None)


def summary_table(path: Path) -> Any:
    import pandas as pd  # gradio dependency

    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["category"] == "ALL"].drop(columns=["category"]).dropna(axis=1, how="all")
    return df.round(3).rename(columns=lambda c: c.replace("_", " "))


def report_names() -> list[str]:
    return sorted(p.name for p in config.REPORTS_DIR.glob("*.md")) if config.REPORTS_DIR.exists() else []


def show_report(name: str | None) -> str:
    if not name or not (config.REPORTS_DIR / name).exists():
        return "_No report yet. Run a benchmark, then press **Regenerate**._"
    return (config.REPORTS_DIR / name).read_text(encoding="utf-8")


def make_reports(asr: str | None, llm: str | None, tts: str | None) -> tuple[Any, str]:
    from voice_lab.benchmarking.report import generate

    picked = {"asr": asr, "llm": llm, "tts": tts}
    names = [p.name for p in generate(runs={k: config.RESULTS_DIR / k / v if v else None for k, v in picked.items()})]
    return gr.update(choices=names, value=names[0]), show_report(names[0])


def run_choices(kind: str, rating: bool = False) -> list[tuple[str, str]]:
    """(label, run dir name), newest first. Labels never name models, so the blind rating tab can use them."""
    root = config.RESULTS_DIR / kind
    out = []
    for run in sorted((p for p in root.glob("*T*Z-*") if p.is_dir()), reverse=True) if root.exists() else []:
        try:
            records = [json.loads(p.read_text(encoding="utf-8")) for p in run.glob("*.json")]
            samples = [s for r in records for s in r["samples"]]
            when = datetime.strptime(run.name[:15], "%Y%m%dT%H%M%S").replace(tzinfo=UTC).astimezone()
        except (OSError, ValueError, KeyError):
            out.append((run.name, run.name))  # unreadable run: still listed, just unlabelled
            continue
        clips = sum(bool(s.get("metrics")) and not s.get("error") for s in samples)
        counts = (
            f"{plural(len(records), 'system')} · {plural(clips, 'clip')}"
            if rating
            else f"{plural(len(records), 'model')} · {plural(len({s['id'] for s in samples}), 'sample')}"
        )
        out.append((f"{when:%Y-%m-%d %H:%M} · {counts}", run.name))
    return out


def tts_runs() -> list[tuple[str, str]]:
    return run_choices("tts", rating=True)


def _rating_view(state: dict[str, Any]) -> tuple[Any, ...]:
    row = state["sheet"][state["i"]]
    prev = state["done"].get(row["clip"], {})
    clip = Path(state["run"]) / "human_eval" / "clips" / f"{row['clip']}.wav"
    n, done = len(state["sheet"]), len(state["done"])
    progress = (
        f'<div class="vl-progress"><div style="width:{100 * done / n:.0f}%"></div></div>'
        f'<p class="vl-hint">Clip <b>{state["i"] + 1}</b> of {n} · {done} rated · Previous and Skip wrap around</p>'
    )
    return (
        str(clip),
        f'<p class="vl-text"><b>Intended text:</b> {esc(row["text"])}</p>',
        progress,
        *[prev.get(c) or None for c in CRITERIA],
        prev.get("comment", ""),
    )


@friendly
def rate_start(run_name: str | None, rater: str) -> tuple[Any, ...]:
    if not run_name:
        raise gr.Error("No TTS run yet: run a TTS benchmark first (Evaluate → Benchmarks).")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", rater.strip()):
        raise gr.Error("Enter an anonymous rater id such as R01 (letters/digits only, never a real name).")
    run_dir = config.RESULTS_DIR / "tts" / run_name
    if not (run_dir / "human_eval" / "key.json").exists():
        make_sheet(run_dir)  # shuffled, anonymised clips; the model key stays in key.json
    with (run_dir / "human_eval" / "ratings_template.csv").open(encoding="utf-8") as f:
        sheet = list(csv.DictReader(f))
    if any(not r["text"] for r in sheet):  # sheets made before sample_texts() existed
        texts = sample_texts(run_dir)
        for r in sheet:
            r["text"] = r["text"] or texts.get(r["sample_id"], "")
    mine = run_dir / "human_eval" / f"ratings_{rater.strip()}.csv"
    done: dict[str, Any] = {}
    if mine.exists():
        with mine.open(encoding="utf-8") as f:
            done = {r["clip"]: r for r in csv.DictReader(f)}
    first_open = next((i for i, r in enumerate(sheet) if r["clip"] not in done), 0)
    state = {"run": str(run_dir), "rater": rater.strip(), "sheet": sheet, "done": done, "i": first_open}
    return state, *_rating_view(state)


@friendly
def rate_save(state: dict[str, Any], *values: Any) -> tuple[Any, ...]:
    if not state:
        raise gr.Error("Press Start first.")
    *scores, comment = values
    if not any(scores) and not (comment or "").strip():
        raise gr.Error("Rate at least one criterion, or press Skip.")
    row = state["sheet"][state["i"]]
    state["done"][row["clip"]] = {
        "clip": row["clip"],
        "sample_id": row["sample_id"],
        "text": row["text"],
        **{c: v or "" for c, v in zip(CRITERIA, scores, strict=True)},
        "comment": comment,
    }
    mine = Path(state["run"]) / "human_eval" / f"ratings_{state['rater']}.csv"
    with mine.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "sample_id", "text", *CRITERIA, "comment"])
        writer.writeheader()
        writer.writerows(state["done"].values())
    import_ratings(Path(state["run"]), mine, state["rater"], config.RESULTS_DIR / "lab.db")  # results -> Reports
    state["i"] = (state["i"] + 1) % len(state["sheet"])
    return state, *_rating_view(state)


def rate_move(state: dict[str, Any], step: int) -> tuple[Any, ...]:
    if not state:
        raise gr.Error("Press Start first.")
    state["i"] = (state["i"] + step) % len(state["sheet"])
    return state, *_rating_view(state)


def tts_examples() -> list[list[str]]:
    try:
        rows = load_manifest(config.ROOT / "data" / "manifests" / "tts.jsonl", "tts")
    except (ManifestError, OSError):
        return []
    picked = {r["category"]: r["text"] for r in rows}  # one example per category
    return [
        [picked[c]]
        for c in ("nepali_english_mixed", "numbers", "currency", "dates", "names", "questions")
        if c in picked
    ]


def asr_examples() -> list[list[str]]:
    folder = config.ROOT / "data" / "synthetic" / "piper-ne-google-x-low"
    return [
        [str(folder / f)] for f in ("p-names-01.wav", "p-mixed-02.wav", "p-numbers-01.wav") if (folder / f).exists()
    ]


# --- Layout ------------------------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    bench = load_benchmark_config()
    profiles = list(bench.get("preprocess_profiles", {}))
    pipe = bench.get("pipeline", {})
    with gr.Blocks(title="Nepali Voice Lab", analytics_enabled=False) as app:
        # Sidebar: runtime settings and system status, visible from every page.
        with gr.Sidebar(label="Lab", open=True, width=300):
            section("Runtime")
            device = gr.Dropdown(
                ["auto", "cpu", "cuda"], value="auto", label="Device", info="auto = CUDA when available, otherwise CPU"
            )
            allow_cloud = gr.Checkbox(
                label="Allow cloud models",
                value=False,
                info="Needed only for Gemini baselines. Sends audio/text to Google.",
                elem_classes=["vl-cloud"],
            )
            section("Models in memory")
            loaded = gr.HTML(loaded_html(), elem_classes=["vl-tight"])
            unload_btn = gr.Button("Unload all models", size="sm")
            section("System")
            side_status = gr.HTML('<p class="vl-hint">Loading…</p>', elem_classes=["vl-tight"])
            refresh_btn = gr.Button("Refresh status", size="sm")

        with gr.Column(elem_classes=["vl-page"]):  # caps the line length on wide screens
            gr.HTML(
                '<div class="vl-header"><div><h1 class="vl-title">Nepali Voice Lab</h1><p class="vl-subtitle">'
                "Benchmark open ASR, LLM and TTS models for Nepali and Nepali-English speech. "
                "Tools are fake and the knowledge base is fictional.</p></div></div>",
                elem_classes=["vl-tight"],
            )
            pills = gr.HTML(elem_classes=["vl-tight"])
            cloud_banner = gr.HTML(
                '<div class="vl-banner warn">Cloud models are allowed. Picking a Gemini model sends '
                "audio or text to Google. Use a personal/test key and never customer data.</div>",
                visible=False,
            )

            with gr.Tabs():
                with gr.Tab("Playground"), gr.Tabs():
                    with gr.Tab("Speech → Text"), gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            section("Input")
                            asr_audio = gr.Audio(
                                sources=["microphone", "upload"],
                                type="filepath",
                                label="Nepali speech",
                                buttons=["download"],
                            )
                            if examples := asr_examples():
                                gr.Examples(examples, inputs=[asr_audio], label="Synthetic sample clips")
                            asr_model = gr.Dropdown(
                                choices("asr"),
                                value=default("asr", pipe.get("asr")),
                                label="ASR model",
                                info="Status shows whether it can run here",
                            )
                            asr_profile = gr.Dropdown(
                                profiles,
                                value="original",
                                label="Preprocessing",
                                info="telephone_8k simulates a phone line (8 kHz, G.711)",
                            )
                            with gr.Row():
                                asr_btn = gr.Button("Transcribe", variant="primary")
                                vad_btn = gr.Button("Detect speech (VAD)")
                            with gr.Accordion("VAD model", open=False):
                                vad_model = gr.Dropdown(
                                    choices("vad"), value=default("vad", pipe.get("vad")), show_label=False
                                )
                        with gr.Column(scale=1):
                            section("Output")
                            asr_text = gr.Textbox(
                                label="Transcript", lines=5, buttons=["copy"], elem_classes=["vl-devanagari"]
                            )
                            asr_stats = gr.HTML(empty_html("Latency, real-time factor and load time appear here."))
                            with gr.Accordion("Speech segments (VAD)", open=False):
                                vad_rows = gr.Dataframe(
                                    headers=["start (s)", "end (s)", "length (s)"], interactive=False
                                )
                                vad_stats = gr.HTML()

                    with gr.Tab("Text → Speech"), gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            section("Input")
                            tts_text = gr.Textbox(
                                label="Text",
                                lines=4,
                                value="नमस्ते, तपाईंको appointment भोलि बिहान १० बजे छ।",
                                elem_classes=["vl-devanagari"],
                            )
                            if examples := tts_examples():
                                gr.Examples(examples, inputs=[tts_text], label="Test sentences")
                            tts_model = gr.Dropdown(
                                choices("tts"), value=default("tts", pipe.get("tts")), label="TTS model"
                            )
                            tts_voice = gr.Textbox(
                                label="Voice (optional)",
                                info="Piper google voices: 0–17 · XTTS: e.g. Claribel Dervla · Gemini: e.g. Kore",
                            )
                            with gr.Accordion("Voice cloning (XTTS only)", open=False):
                                tts_ref = gr.Audio(
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                    label="Reference voice - use only with the speaker's consent",
                                    buttons=["download"],
                                )
                            tts_btn = gr.Button("Speak", variant="primary")
                        with gr.Column(scale=1):
                            section("Output")
                            tts_audio = gr.Audio(
                                label="Speech", type="filepath", interactive=False, buttons=["download"]
                            )
                            tts_stats = gr.HTML(empty_html("Latency, real-time factor and sample rate appear here."))

                    with gr.Tab("Chat"):
                        chat_state = gr.State({"turns": [], "display": []})
                        with gr.Row():
                            llm_model = gr.Dropdown(
                                choices("llm"), value=default("llm", pipe.get("llm")), label="LLM", scale=3
                            )
                            llm_tools = gr.Checkbox(
                                label="Fake tools", value=True, scale=1, info="Customer, balance, appointments"
                            )
                        gr.HTML(
                            '<p class="vl-hint">The first reply also loads the model into Ollama (2–3 min on this '
                            "CPU); later replies are faster.</p>",
                            elem_classes=["vl-tight"],
                        )
                        with gr.Accordion("System prompt", open=False):
                            llm_system = gr.Textbox(value=system_prompt(), lines=5, show_label=False)
                        chatbot = gr.Chatbot(
                            height="55vh",
                            label="Conversation",
                            buttons=["copy"],
                            placeholder="Ask in Nepali, English or both. Tool calls appear inline.",
                            elem_classes=["vl-devanagari"],
                        )
                        with gr.Row(equal_height=True):
                            chat_in = gr.Textbox(
                                placeholder="मेरो balance कति छ? Customer ID C-1001",
                                show_label=False,
                                scale=6,
                                lines=1,
                                max_lines=4,
                                elem_classes=["vl-devanagari"],
                            )
                            chat_btn = gr.Button("Send", variant="primary", scale=1)
                            chat_clear = gr.Button("Clear", scale=1)
                        gr.Examples([[e] for e in CHAT_EXAMPLES], inputs=[chat_in], label="Try")
                        chat_stats = gr.HTML()

                    with gr.Tab("Conversation"), gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            section("Input", "Speak → VAD → ASR → LLM → TTS. Recorded audio only, not a phone call.")
                            conv_audio = gr.Audio(
                                sources=["microphone", "upload"],
                                type="filepath",
                                label="Your question",
                                buttons=["download"],
                            )
                            with gr.Row():
                                conv_vad = gr.Dropdown(
                                    [*choices("vad"), NO_VAD], value=default("vad", pipe.get("vad")), label="VAD"
                                )
                                conv_asr = gr.Dropdown(
                                    choices("asr"), value=default("asr", pipe.get("asr")), label="ASR"
                                )
                            with gr.Row():
                                conv_llm = gr.Dropdown(
                                    choices("llm"), value=default("llm", pipe.get("llm")), label="LLM"
                                )
                                conv_tts = gr.Dropdown(
                                    choices("tts"), value=default("tts", pipe.get("tts")), label="TTS"
                                )
                            conv_tools = gr.Checkbox(label="Fake tools", value=True)
                            conv_btn = gr.Button("Run conversation", variant="primary")
                        with gr.Column(scale=1):
                            section("Output")
                            conv_transcript = gr.Textbox(label="Heard (ASR)", lines=2, elem_classes=["vl-devanagari"])
                            conv_reply = gr.Textbox(label="Reply (LLM)", lines=3, elem_classes=["vl-devanagari"])
                            conv_out = gr.Audio(
                                label="Spoken reply (TTS)", type="filepath", interactive=False, buttons=["download"]
                            )
                            conv_details = gr.HTML(empty_html("A timing breakdown per stage appears here."))

                with gr.Tab("Evaluate"), gr.Tabs():
                    with gr.Tab("Benchmarks"), gr.Row(equal_height=False):
                        with gr.Column(scale=2):
                            section("Configuration", "Finished samples are cached; tick Re-run to force them again.")
                            b_kind = gr.Radio(
                                [("ASR", "asr"), ("Telephone ASR", "telephone"), ("LLM", "llm"), ("TTS", "tts")],
                                value="asr",
                                label="Benchmark",
                            )
                            b_models = gr.Dropdown(choices("asr"), multiselect=True, label="Models")
                            b_dataset = gr.Textbox(value=default_dataset("asr"), label="Dataset manifest")
                            with gr.Row():
                                b_limit = gr.Number(
                                    value=None, label="Limit", precision=0, info="First N samples; empty = all"
                                )
                                b_force = gr.Checkbox(label="Re-run cached samples")
                            b_profiles = gr.Dropdown(
                                profiles,
                                multiselect=True,
                                label="Preprocessing profiles",
                                info="Empty = the default from configs/benchmark.yaml",
                            )
                            b_judge = gr.Dropdown(
                                choices("asr"),
                                value=None,
                                label="Intelligibility judge (ASR model)",
                                visible=False,
                                info="Optional: transcribes the TTS output to estimate CER",
                            )
                            with gr.Row():
                                b_run = gr.Button("Run benchmark", variant="primary", scale=3)
                                b_stop = gr.Button("Stop", variant="stop", scale=1)
                        with gr.Column(scale=3):
                            section("Results")
                            b_status = gr.HTML(empty_html("Configure a run on the left, then press Run benchmark."))
                            b_table = gr.Dataframe(
                                label="Summary (all samples)", interactive=False, wrap=True, visible=False
                            )
                            with gr.Accordion("Log", open=False):
                                b_log = gr.Textbox(show_label=False, lines=14, max_lines=30, elem_classes=["vl-log"])

                    with gr.Tab("Reports"):
                        section("Runs", "Reports are built from the selected runs.")
                        with gr.Row(equal_height=True):
                            r_runs = [
                                gr.Dropdown(runs, value=runs[0][1] if runs else None, label=f"{k.upper()} run", scale=2)
                                for k, runs in [(k, run_choices(k)) for k in ("asr", "llm", "tts")]
                            ]
                            r_make = gr.Button("Regenerate", variant="primary", scale=1)
                        r_pick = gr.Dropdown(report_names(), value=next(iter(report_names()), None), label="Report")
                        r_view = gr.Markdown(show_report(next(iter(report_names()), None)), elem_classes=["vl-report"])

                    with gr.Tab("Human rating"):
                        section(
                            "Blind 1–5 rating of TTS clips",
                            "Model names are hidden and clips are shuffled. Use a pseudonym (R01, R02…), never "
                            "your name. Leave a criterion empty if it does not apply.",
                        )
                        gr.HTML(f'<p class="vl-hint"><b>Scale:</b> {esc(SCALE)}</p>', elem_classes=["vl-tight"])
                        with gr.Accordion("What each criterion means", open=False):
                            gr.HTML(
                                "".join(
                                    f'<p class="vl-hint"><b>{esc(c.replace("_", " ").capitalize())}:</b> '
                                    f"{esc(CRITERIA_HELP[c])}</p>"
                                    for c in CRITERIA
                                ),
                                elem_classes=["vl-tight"],
                            )
                        h_state = gr.State(None)
                        with gr.Row(equal_height=True):
                            runs = tts_runs()
                            h_run = gr.Dropdown(
                                runs, value=runs[0][1] if runs else None, label="TTS benchmark run", scale=3
                            )
                            h_rater = gr.Textbox(value="R01", label="Rater id", scale=1)
                            h_start = gr.Button("Start / resume", variant="primary", scale=1)
                        h_progress = gr.HTML()
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=2):
                                h_audio = gr.Audio(
                                    type="filepath", interactive=False, label="Clip", buttons=["download"]
                                )
                                h_text = gr.HTML()
                            with gr.Column(scale=3):
                                with gr.Group(elem_classes=["vl-criteria"]):
                                    h_scores = [
                                        gr.Radio(["1", "2", "3", "4", "5"], label=c.replace("_", " ").capitalize())
                                        for c in CRITERIA
                                    ]
                                h_comment = gr.Textbox(label="Comment (optional)")
                                with gr.Row():
                                    h_prev = gr.Button("Previous")
                                    h_save = gr.Button("Save & next", variant="primary")
                                    h_next = gr.Button("Skip")
                        gr.HTML(
                            '<p class="vl-hint">Results stay hidden while you rate. When done, open Evaluate → '
                            "Reports and press Regenerate: tts-comparison.md shows the per-model means.</p>",
                            elem_classes=["vl-tight"],
                        )

                with gr.Tab("Models"):
                    with gr.Row(equal_height=True):
                        m_kind = gr.Radio(
                            [("All", "all"), ("ASR", "asr"), ("LLM", "llm"), ("TTS", "tts"), ("VAD", "vad")],
                            value="all",
                            label="Kind",
                            scale=3,
                        )
                        m_ready = gr.Checkbox(label="Only models that can run here", scale=2)
                    models_df = gr.Dataframe(
                        headers=["Kind", "Model", "Provider", "Status", "Commercial use", "Size", "Default"],
                        datatype=["str", "str", "str", "html", "html", "str", "str"],
                        interactive=False,
                        wrap=True,
                        show_search="search",
                        pinned_columns=2,
                    )
                    with gr.Group():
                        section(
                            "Download a model", "Nothing downloads automatically. Check the size and licence first."
                        )
                        with gr.Row(equal_height=True):
                            dl_model = gr.Dropdown(downloadable(), value=None, label="Model", scale=3)
                            dl_check = gr.Button("Show size & licence", scale=1)
                        with gr.Row(equal_height=True):
                            dl_ok = gr.Checkbox(label="I checked the size and licence", scale=3)
                            dl_go = gr.Button("Download", variant="primary", interactive=False, scale=1)
                        dl_log = gr.Textbox(label="Output", lines=6, max_lines=16, elem_classes=["vl-log"])
                    with gr.Accordion("System details", open=False):
                        sys_json = gr.JSON(show_label=False)

        # --- Events ----------------------------------------------------------------------------------
        model_dropdowns = [
            (asr_model, "asr"),
            (vad_model, "vad"),
            (tts_model, "tts"),
            (llm_model, "llm"),
            (conv_vad, "vad"),
            (conv_asr, "asr"),
            (conv_llm, "llm"),
            (conv_tts, "tts"),
            (b_judge, "asr"),
        ]

        def refresh_all(kind: str, only_ready: bool) -> tuple[Any, ...]:
            """Statuses and every list that can change while the UI runs (downloads, new runs, reports)."""
            refresh_status()
            info = system_state()
            runs = [tts_runs(), *[run_choices(k) for k in ("asr", "llm", "tts")]]
            return (
                status_pills(info),
                sidebar_html(info),
                info,
                models_rows(kind, only_ready),
                gr.update(choices=downloadable()),
                gr.update(choices=["auto", "cpu", "cuda"] if info["gpus"] else ["auto", "cpu"]),
                *[gr.update(choices=[*choices(k), *([NO_VAD] if d is conv_vad else [])]) for d, k in model_dropdowns],
                *[gr.update(choices=r, value=r[0][1] if r else None) for r in runs],
                gr.update(choices=report_names()),
            )

        refresh_outputs = [
            pills,
            side_status,
            sys_json,
            models_df,
            dl_model,
            device,
            *[d for d, _ in model_dropdowns],
            h_run,
            *r_runs,
            r_pick,
        ]
        app.load(refresh_all, [m_kind, m_ready], refresh_outputs)
        app.load(on_benchmark_kind, b_kind, [b_models, b_dataset, b_profiles, b_judge])
        refresh_btn.click(refresh_all, [m_kind, m_ready], refresh_outputs)
        unload_btn.click(unload, outputs=loaded)
        allow_cloud.change(lambda on: gr.update(visible=on), allow_cloud, cloud_banner)

        def with_loaded(event: Any) -> None:
            event.then(loaded_html, outputs=loaded)

        with_loaded(
            asr_btn.click(
                transcribe,
                [asr_audio, asr_model, asr_profile, device, allow_cloud],
                [asr_text, asr_stats],
                api_name="transcribe",
                show_progress_on=asr_text,  # one loading/error indicator, not one per output
                **HEAVY,
            )
        )
        with_loaded(
            vad_btn.click(
                detect,
                [asr_audio, vad_model, device, allow_cloud],
                [vad_rows, vad_stats],
                api_name="vad",
                show_progress_on=vad_rows,
                **HEAVY,
            )
        )
        with_loaded(
            tts_btn.click(
                speak,
                [tts_text, tts_model, tts_voice, tts_ref, device, allow_cloud],
                [tts_audio, tts_stats],
                api_name="speak",
                show_progress_on=tts_audio,
                **HEAVY,
            )
        )
        chat_args = [chat_in, chat_state, llm_model, llm_tools, llm_system, device, allow_cloud]
        chat_out = [chatbot, chat_state, chat_in, chat_stats]
        with_loaded(chat_btn.click(chat, chat_args, chat_out, api_name="chat", **HEAVY))
        with_loaded(chat_in.submit(chat, chat_args, chat_out, **HEAVY))
        chat_clear.click(clear_chat, outputs=[chatbot, chat_state, chat_stats])
        with_loaded(
            conv_btn.click(
                converse_ui,
                [conv_audio, conv_vad, conv_asr, conv_llm, conv_tts, conv_tools, device, allow_cloud],
                [conv_transcript, conv_reply, conv_out, conv_details],
                api_name="conversation",
                **HEAVY,
            )
        )

        b_kind.change(on_benchmark_kind, b_kind, [b_models, b_dataset, b_profiles, b_judge])
        run_event = b_run.click(
            run_benchmark_ui,
            [b_kind, b_models, b_dataset, b_limit, b_force, b_profiles, b_judge, device, allow_cloud],
            [b_status, b_log, b_table],
            **HEAVY,
        )
        run_event.then(refresh_all, [m_kind, m_ready], refresh_outputs)
        b_stop.click(lambda: pill("Stopped", "warn"), outputs=b_status, cancels=[run_event])
        r_pick.change(show_report, r_pick, r_view)
        r_make.click(make_reports, r_runs, [r_pick, r_view])

        view = [h_audio, h_text, h_progress, *h_scores, h_comment]
        h_start.click(rate_start, [h_run, h_rater], [h_state, *view])
        h_save.click(rate_save, [h_state, *h_scores, h_comment], [h_state, *view])
        h_prev.click(lambda s: rate_move(s, -1), h_state, [h_state, *view])
        h_next.click(lambda s: rate_move(s, 1), h_state, [h_state, *view])

        m_kind.change(models_rows, [m_kind, m_ready], models_df)
        m_ready.change(models_rows, [m_kind, m_ready], models_df)
        dl_ok.change(lambda ok: gr.update(interactive=ok), dl_ok, dl_go)
        dl_check.click(check_download, dl_model, dl_log)
        dl_go.click(do_download, [dl_model, dl_ok], dl_log, **HEAVY).then(
            refresh_all, [m_kind, m_ready], refresh_outputs
        )
    return app


def launch(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = False) -> None:
    app = build_app()
    app.queue(default_concurrency_limit=1)  # one heavy model job at a time on a small box
    app.launch(
        server_name=host,
        server_port=port,
        share=False,
        inbrowser=open_browser,
        theme=THEME,
        css_paths=[CSS],
        footer_links=["api", "settings"],
        allowed_paths=[str(config.RESULTS_DIR), str(config.ROOT / "data")],
    )
