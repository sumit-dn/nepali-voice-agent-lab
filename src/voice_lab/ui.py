"""Local web UI (Gradio) over the same providers the CLI uses. Start with `voice-lab ui`.

Local by construction: binds to 127.0.0.1, no share link, Gradio analytics off, system fonts (no Google Fonts),
weights load only from the local cache, and cloud models still need the "allow cloud" switch.
Benchmarks and downloads run the real `voice-lab` CLI in a subprocess, so their logic is not duplicated here.
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # must precede `import gradio`
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never fetch weights implicitly (must precede HF imports)

import csv
import gc
import json
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
from voice_lab.evaluation.human import CRITERIA, import_ratings, make_sheet
from voice_lab.evaluation.manifest import ManifestError
from voice_lab.llm.tools import run_with_tools
from voice_lab.providers.base import Provider, ProviderError
from voice_lab.providers.registry import build

READY = {"downloaded", "pulled", "bundled with pip pkg"}
SILENCE_DBFS = -60.0  # quieter than this is treated as "the mic recorded nothing"
USER_ERRORS = (ConfigError, ProviderError, ManifestError, ValueError, FileNotFoundError)
NOISE = ("FRAME_DURATION_MS", "Missing phoneme", "warnings.warn", "Warning:", "Fetching ")
THEME = gr.themes.Soft(font=["system-ui", "sans-serif"], font_mono=["monospace"])

# ponytail: one global model cache + queue concurrency 1 (one heavy job at a time); fine for a single local
# user on a 16 GB box, needs per-session limits before anyone shares this UI.
_providers: dict[tuple[Any, ...], Provider] = {}
_lock = threading.Lock()


def friendly(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Show expected failures (not downloaded, server down, bad input) as a UI message, not a traceback."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except USER_ERRORS as e:
            raise gr.Error(str(e)) from e

    return wrapper


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
    with _lock:
        n = len(_providers)
        _providers.clear()
    gc.collect()
    return f"Unloaded {n} model(s) from memory."


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


# --- System & models -----------------------------------------------------------------------------


def system_markdown() -> str:
    from voice_lab.config import OPTIONAL_ENV, missing_credentials
    from voice_lab.utils.hardware import system_info
    from voice_lab.utils.http import get_json

    info = system_info()
    gpus = ", ".join(f"{g['name']} ({g['vram_gb']} GB)" for g in info["gpus"]) or "none → CPU-only"
    root = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        ollama = f"v{get_json(f'{root}/api/version', timeout=3).get('version')} at {root}"
    except ProviderError:
        ollama = f"not reachable at {root}"
    missing = missing_credentials()
    creds = ", ".join(f"`{k}` {'✗' if k in missing else '✓'}" for k in OPTIONAL_ENV)
    pkgs = ", ".join(f"{k} {v}" for k, v in info["packages"].items() if v)
    return (
        f"| | |\n|---|---|\n| CPU | {info['cpu']} ({info['cpu_threads']} threads) |\n"
        f"| RAM | {info['ram_total_gb']} GB total, {info['ram_available_gb']} GB free |\n"
        f"| GPU | {gpus} |\n| Disk free | {info['disk_free_gb']} GB |\n| Ollama | {ollama} |\n"
        f"| Packages | {pkgs} |\n| Optional keys | {creds} (none needed for local models) |"
    )


def models_rows() -> list[list[Any]]:
    return [
        [
            kind,
            m.id,
            m.provider,
            "on" if m.enabled else "off",
            "yes" if m.external else "",
            m.commercial_use,
            m.size or "",
            local_status(m),
        ]
        for kind in KINDS
        for m in load_models(kind).values()
    ]


def refresh_system() -> tuple[str, list[list[Any]]]:
    refresh_status()
    return system_markdown(), models_rows()


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
    yield log
    for line in proc.stdout:
        if not any(n in line for n in NOISE):
            log += line
            yield log
    proc.wait()
    yield log + f"\n[finished, exit code {proc.returncode}]"


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


# --- Interactive tabs -----------------------------------------------------------------------------


@friendly
def transcribe(audio: str | None, model_id: str, profile: str, device: str, allow_cloud: bool) -> tuple[str, str]:
    audio = require_sound(audio)
    steps = load_benchmark_config()["preprocess_profiles"][profile]
    p = provider("asr", model_id, device, allow_cloud)
    r = p.transcribe(load_and_prepare(audio, steps))
    stats = (
        f"**{r.model}** · {r.latency_seconds:.2f} s for {r.audio_duration_seconds:.2f} s of audio · "
        f"RTF {r.real_time_factor:.2f} · load {p.load_seconds:.1f} s · profile `{profile}`"
    )
    if conversion := r.metadata.get("model_input_conversion"):
        stats += f" · model input: {', '.join(conversion)}"
    return r.text, stats


@friendly
def detect(audio: str | None, model_id: str, device: str, allow_cloud: bool) -> tuple[list[list[float]], str]:
    audio = require_sound(audio)
    r = provider("vad", model_id, device, allow_cloud).detect_speech(load_audio(audio))
    rows = [[round(s, 2), round(e, 2), round(e - s, 2)] for s, e in r.segments]
    return rows, (
        f"{len(rows)} speech segment(s), {r.speech_seconds:.2f} s of {r.audio_duration_seconds:.2f} s · "
        f"{r.latency_seconds * 1000:.0f} ms"
    )


@friendly
def speak(
    text: str, model_id: str, voice: str, reference: str | None, device: str, allow_cloud: bool
) -> tuple[str, str]:
    if not text.strip():
        raise gr.Error("Type some text first.")
    if reference:
        voice = require_sound(reference)  # XTTS zero-shot cloning from an uploaded/recorded clip
    p = provider("tts", model_id, device, allow_cloud)
    r = p.synthesize(text, out_wav(model_id), voice=voice.strip() or None)
    return r.audio_path, (
        f"**{r.model}** · {r.latency_seconds:.2f} s for {r.audio_duration_seconds:.2f} s of audio · "
        f"RTF {r.real_time_factor:.2f} · {r.sample_rate} Hz · load {p.load_seconds:.1f} s · "
        f"saved to `{r.audio_path}`"
    )


def _tool_messages(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": f"`{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})` → "
            f"`{json.dumps(c['result'], ensure_ascii=False)}`",
            "metadata": {"title": f"🔧 fake tool: {c['name']}"},
        }
        for c in trace
    ]


@friendly
def chat(
    message: str, state: dict[str, Any], model_id: str, tools: bool, system: str, device: str, allow_cloud: bool
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    """`state` keeps the model-facing turns separate from the displayed chat (which also shows tool calls)."""
    state = state or {"turns": [], "display": []}
    if not message.strip():
        return state["display"], state, "", ""
    llm = provider("llm", model_id, device, allow_cloud)
    turns = [*state["turns"], {"role": "user", "content": message}]
    messages = ([{"role": "system", "content": system}] if system.strip() else []) + turns
    if tools:
        result, trace, ttft, seconds = run_with_tools(llm, messages)
    else:
        result = llm.generate(messages)
        trace, ttft, seconds = [], result.time_to_first_token, result.latency_seconds
    reply = result.text or "(empty reply)"
    display = [
        *state["display"],
        {"role": "user", "content": message},
        *_tool_messages(trace),
        {"role": "assistant", "content": reply},
    ]
    state = {"turns": [*turns, {"role": "assistant", "content": reply}], "display": display}
    stats = (
        f"**{result.model}** · first token {ttft:.2f} s · total {seconds:.2f} s · {result.output_tokens} output tokens"
        if ttft is not None
        else f"total {seconds:.2f} s"
    )
    return display, state, "", stats


def clear_chat() -> tuple[list[Any], dict[str, Any], str]:
    return [], {"turns": [], "display": []}, ""


@friendly
def converse_ui(
    audio: str | None, vad_id: str, asr_id: str, llm_id: str, tts_id: str, tools: bool, device: str, allow_cloud: bool
) -> tuple[str, str, str | None, str]:
    from voice_lab.pipeline import converse, waterfall

    audio = require_sound(audio)
    r = converse(
        audio,
        vad=provider("vad", vad_id, device, allow_cloud) if vad_id else None,
        asr=provider("asr", asr_id, device, allow_cloud),
        llm=provider("llm", llm_id, device, allow_cloud),
        tts=provider("tts", tts_id, device, allow_cloud),
        out_dir=config.RESULTS_DIR / "ui" / f"conversation-{datetime.now(UTC):%Y%m%dT%H%M%S%f}",
        system_prompt=system_prompt(),
        tools=tools,
    )
    details = f"```\n{waterfall(r['timings'])}\n```\nModel load times (excluded above): {r['load_seconds']}"
    if r.get("tool_calls"):
        details += "\n\n" + "\n".join(
            f"- {m['metadata']['title']}: {m['content']}" for m in _tool_messages(r["tool_calls"])
        )
    if r.get("error"):
        details = f"**{r['error']}**\n\n{details}"
    return r.get("transcript", ""), r.get("response", ""), r.get("response_audio"), details


# --- Benchmarks & reports --------------------------------------------------------------------------


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
        gr.update(visible=kind == "tts"),
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
) -> Iterator[tuple[str, Any]]:
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
        yield log, None
    found = re.search(r"^Results: (.+)$", log, re.M)
    yield log, summary_table(Path(found.group(1)) / "summary.csv") if found else None


def summary_table(path: Path) -> Any:
    import pandas as pd  # gradio dependency

    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["category"] == "ALL"].drop(columns=["category"]).dropna(axis=1, how="all")
    return df.round(3)


def report_names() -> list[str]:
    return sorted(p.name for p in config.REPORTS_DIR.glob("*.md")) if config.REPORTS_DIR.exists() else []


def show_report(name: str | None) -> str:
    return (config.REPORTS_DIR / name).read_text(encoding="utf-8") if name else "_No report selected._"


def make_reports() -> tuple[Any, str]:
    from voice_lab.benchmarking.report import generate

    names = [p.name for p in generate()]
    return gr.update(choices=names, value=names[0]), show_report(names[0])


# --- Human rating ----------------------------------------------------------------------------------


def tts_runs() -> list[str]:
    root = config.RESULTS_DIR / "tts"
    return sorted((p.name for p in root.glob("*T*Z-*") if p.is_dir()), reverse=True) if root.exists() else []


def _rating_view(state: dict[str, Any]) -> tuple[Any, ...]:
    row = state["sheet"][state["i"]]
    prev = state["done"].get(row["clip"], {})
    clip = Path(state["run"]) / "human_eval" / "clips" / f"{row['clip']}.wav"
    progress = f"Clip **{state['i'] + 1}** of {len(state['sheet'])} · you have rated {len(state['done'])}"
    return (
        str(clip),
        f"**Intended text:** {row['text']}",
        progress,
        *[prev.get(c) or None for c in CRITERIA],
        prev.get("comment", ""),
    )


@friendly
def rate_start(run_name: str | None, rater: str) -> tuple[Any, ...]:
    if not run_name:
        raise gr.Error("No TTS run yet: run a TTS benchmark first (Benchmarks tab).")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", rater.strip()):
        raise gr.Error("Enter an anonymous rater id such as R01 (letters/digits only, never a real name).")
    run_dir = config.RESULTS_DIR / "tts" / run_name
    if not (run_dir / "human_eval" / "key.json").exists():
        make_sheet(run_dir)  # shuffled, anonymised clips; the model key stays in key.json
    with (run_dir / "human_eval" / "ratings_template.csv").open(encoding="utf-8") as f:
        sheet = list(csv.DictReader(f))
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
    summary = import_ratings(Path(state["run"]), mine, state["rater"], config.RESULTS_DIR / "lab.db")
    state["i"] = (state["i"] + 1) % len(state["sheet"])
    return state, *_rating_view(state), summary


def rate_move(state: dict[str, Any], step: int) -> tuple[Any, ...]:
    if not state:
        raise gr.Error("Press Start first.")
    state["i"] = (state["i"] + step) % len(state["sheet"])
    return state, *_rating_view(state)


# --- Layout ----------------------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    bench = load_benchmark_config()
    profiles = list(bench.get("preprocess_profiles", {}))
    pipe = bench.get("pipeline", {})
    with gr.Blocks(title="Nepali Voice Lab", analytics_enabled=False) as app:
        gr.Markdown(
            "# Nepali Voice Lab\nLocal research UI. Nothing leaves this machine unless you tick "
            "*Allow cloud models* **and** pick a cloud model. Tools are fake; the knowledge base is fictional."
        )
        with gr.Row():
            device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto", label="Device", scale=1)
            allow_cloud = gr.Checkbox(
                label="Allow cloud models (sends audio/text to Google Gemini)", value=False, scale=2
            )
            unload_btn = gr.Button("Unload models from memory", scale=1)
            unload_msg = gr.Markdown()
        unload_btn.click(unload, outputs=unload_msg)

        with gr.Tab("System & models"):
            sys_md = gr.Markdown("Loading…")
            refresh_btn = gr.Button("Refresh")
            models_df = gr.Dataframe(
                headers=["kind", "id", "provider", "enabled", "cloud", "commercial use", "size", "status"],
                interactive=False,
                wrap=True,
            )
            gr.Markdown("### Download a model\nNothing downloads automatically. Check the size and licence first.")
            with gr.Row():
                dl_model = gr.Dropdown(
                    [m[1] for m in models_rows() if m[2] not in ("none", "silero") and not m[4]], label="Model", scale=2
                )
                dl_check = gr.Button("Show size & licence", scale=1)
                dl_ok = gr.Checkbox(label="I checked the size and licence", scale=1)
                dl_go = gr.Button("Download", variant="primary", scale=1)
            dl_log = gr.Textbox(label="Output", lines=8, max_lines=20)
            refresh_btn.click(refresh_system, outputs=[sys_md, models_df])
            dl_check.click(check_download, inputs=dl_model, outputs=dl_log)
            dl_go.click(do_download, inputs=[dl_model, dl_ok], outputs=dl_log)

        with gr.Tab("Speech → Text"):
            with gr.Row():
                with gr.Column():
                    asr_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Nepali speech")
                    asr_model = gr.Dropdown(choices("asr"), value=default("asr", pipe.get("asr")), label="ASR model")
                    asr_profile = gr.Dropdown(
                        profiles, value="original", label="Preprocessing (telephone_8k = phone line)"
                    )
                    asr_btn = gr.Button("Transcribe", variant="primary")
                with gr.Column():
                    asr_text = gr.Textbox(label="Transcript", lines=4)
                    asr_stats = gr.Markdown()
                    vad_model = gr.Dropdown(choices("vad"), value=default("vad", pipe.get("vad")), label="VAD model")
                    vad_btn = gr.Button("Detect speech segments (VAD)")
                    vad_rows = gr.Dataframe(headers=["start (s)", "end (s)", "length (s)"], interactive=False)
                    vad_stats = gr.Markdown()
            asr_btn.click(
                transcribe,
                [asr_audio, asr_model, asr_profile, device, allow_cloud],
                [asr_text, asr_stats],
                api_name="transcribe",
            )
            vad_btn.click(detect, [asr_audio, vad_model, device, allow_cloud], [vad_rows, vad_stats], api_name="vad")

        with gr.Tab("Text → Speech"):
            with gr.Row():
                with gr.Column():
                    tts_text = gr.Textbox(label="Text", lines=3, value="नमस्ते, तपाईंको appointment भोलि बिहान १० बजे छ।")
                    tts_model = gr.Dropdown(choices("tts"), value=default("tts", pipe.get("tts")), label="TTS model")
                    tts_voice = gr.Textbox(
                        label="Voice / speaker id (optional)",
                        placeholder="Piper google: 0-17 · XTTS: e.g. Claribel Dervla · Gemini: e.g. Kore",
                    )
                    tts_ref = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="XTTS only: reference voice to clone (use only with the speaker's consent)",
                    )
                    tts_btn = gr.Button("Speak", variant="primary")
                with gr.Column():
                    tts_audio = gr.Audio(label="Output", type="filepath", interactive=False)
                    tts_stats = gr.Markdown()
            tts_btn.click(
                speak,
                [tts_text, tts_model, tts_voice, tts_ref, device, allow_cloud],
                [tts_audio, tts_stats],
                api_name="speak",
            )

        with gr.Tab("Chat (LLM)"):
            chat_state = gr.State({"turns": [], "display": []})
            with gr.Row():
                llm_model = gr.Dropdown(choices("llm"), value=default("llm", pipe.get("llm")), label="LLM", scale=2)
                llm_tools = gr.Checkbox(
                    label="Enable fake tools (customer, balance, appointments)", value=True, scale=2
                )
            with gr.Accordion("System prompt", open=False):
                llm_system = gr.Textbox(value=system_prompt(), lines=5, show_label=False)
            chatbot = gr.Chatbot(height=420, label="Conversation")
            chat_in = gr.Textbox(placeholder="मेरो balance कति छ? Customer ID C-1001", label="Message")
            with gr.Row():
                chat_btn = gr.Button("Send", variant="primary")
                chat_clear = gr.Button("Clear")
            chat_stats = gr.Markdown("On this CPU an 8B model takes ~5–40 s per reply.")
            chat_args = [chat_in, chat_state, llm_model, llm_tools, llm_system, device, allow_cloud]
            chat_out = [chatbot, chat_state, chat_in, chat_stats]
            chat_btn.click(chat, chat_args, chat_out, api_name="chat")
            chat_in.submit(chat, chat_args, chat_out)
            chat_clear.click(clear_chat, outputs=[chatbot, chat_state, chat_stats])

        with gr.Tab("Conversation"):
            gr.Markdown("Speak → VAD → ASR → LLM → TTS. Recorded audio only; this is not a phone integration.")
            with gr.Row():
                with gr.Column():
                    conv_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Your question")
                    conv_vad = gr.Dropdown(choices("vad"), value=default("vad", pipe.get("vad")), label="VAD")
                    conv_asr = gr.Dropdown(choices("asr"), value=default("asr", pipe.get("asr")), label="ASR")
                    conv_llm = gr.Dropdown(choices("llm"), value=default("llm", pipe.get("llm")), label="LLM")
                    conv_tts = gr.Dropdown(choices("tts"), value=default("tts", pipe.get("tts")), label="TTS")
                    conv_tools = gr.Checkbox(label="Enable fake tools", value=True)
                    conv_btn = gr.Button("Run conversation", variant="primary")
                with gr.Column():
                    conv_transcript = gr.Textbox(label="Heard (ASR)", lines=2)
                    conv_reply = gr.Textbox(label="Reply (LLM)", lines=3)
                    conv_out = gr.Audio(label="Spoken reply (TTS)", type="filepath", interactive=False)
                    conv_details = gr.Markdown()
            conv_btn.click(
                converse_ui,
                [conv_audio, conv_vad, conv_asr, conv_llm, conv_tts, conv_tools, device, allow_cloud],
                [conv_transcript, conv_reply, conv_out, conv_details],
                api_name="conversation",
            )

        with gr.Tab("Benchmarks"):
            with gr.Row():
                b_kind = gr.Radio(["asr", "telephone", "llm", "tts"], value="asr", label="Benchmark")
                b_limit = gr.Number(label="Limit (first N samples, empty = all)", precision=0)
                b_force = gr.Checkbox(label="Re-run cached samples (--force)")
            b_models = gr.Dropdown(choices("asr"), multiselect=True, label="Models")
            b_dataset = gr.Textbox(value=default_dataset("asr"), label="Dataset manifest")
            b_profiles = gr.Dropdown(profiles, multiselect=True, label="ASR preprocessing profiles (empty = default)")
            b_judge = gr.Dropdown(
                choices("asr"), label="TTS: ASR model as intelligibility judge (optional)", visible=False
            )
            b_run = gr.Button("Run benchmark", variant="primary")
            b_log = gr.Textbox(label="Log", lines=10, max_lines=25)
            b_table = gr.Dataframe(label="Summary (all samples)", interactive=False, wrap=True)
            b_kind.change(on_benchmark_kind, b_kind, [b_models, b_dataset, b_profiles, b_judge])
            b_run.click(
                run_benchmark_ui,
                [b_kind, b_models, b_dataset, b_limit, b_force, b_profiles, b_judge, device, allow_cloud],
                [b_log, b_table],
            )

        with gr.Tab("Reports"):
            with gr.Row():
                r_pick = gr.Dropdown(report_names(), label="Report", scale=3)
                r_make = gr.Button("Regenerate from latest results", scale=1)
            r_view = gr.Markdown()
            r_pick.change(show_report, r_pick, r_view)
            r_make.click(make_reports, outputs=[r_pick, r_view])

        with gr.Tab("Human rating"):
            gr.Markdown(
                "Blind 1–5 rating of TTS clips. Model names are hidden and clips are shuffled. "
                "Use a pseudonym (R01, R02…), never your name. Leave a criterion empty if it does not apply."
            )
            h_state = gr.State(None)
            with gr.Row():
                h_run = gr.Dropdown(tts_runs(), value=next(iter(tts_runs()), None), label="TTS benchmark run")
                h_rater = gr.Textbox(value="R01", label="Rater id")
                h_start = gr.Button("Start / resume", variant="primary")
            h_progress = gr.Markdown()
            h_audio = gr.Audio(type="filepath", interactive=False, label="Clip")
            h_text = gr.Markdown()
            with gr.Row():
                h_scores = [gr.Radio(["1", "2", "3", "4", "5"], label=c.replace("_", " ")) for c in CRITERIA]
            h_comment = gr.Textbox(label="Comment (optional)")
            with gr.Row():
                h_prev = gr.Button("◀ Previous")
                h_save = gr.Button("Save & next ▶", variant="primary")
                h_next = gr.Button("Skip ▶")
            h_summary = gr.JSON(label="Your ratings so far (mean ± 95% CI per system)")
            view = [h_audio, h_text, h_progress, *h_scores, h_comment]
            h_start.click(rate_start, [h_run, h_rater], [h_state, *view])
            h_save.click(rate_save, [h_state, *h_scores, h_comment], [h_state, *view, h_summary])
            h_prev.click(lambda s: rate_move(s, -1), h_state, [h_state, *view])
            h_next.click(lambda s: rate_move(s, 1), h_state, [h_state, *view])

        app.load(refresh_system, outputs=[sys_md, models_df])
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
        allowed_paths=[str(config.RESULTS_DIR)],
    )
