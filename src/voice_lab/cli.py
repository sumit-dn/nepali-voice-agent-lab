"""voice-lab command line.

Everything runs locally. Nothing is downloaded except by `voice-lab download`, and nothing leaves the
machine unless a model is marked `external: true` AND --allow-cloud is passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _find_model(model_id: str, kind: str | None = None) -> Any:
    from voice_lab.config import KINDS, ConfigError, load_models

    for k in [kind] if kind else KINDS:
        if model_id in (models := load_models(k)):
            return models[model_id]
    raise ConfigError(f"Unknown model {model_id!r}. List them with: voice-lab models")


def _build(kind: str, model_id: str, args: argparse.Namespace) -> Any:
    from voice_lab.providers.registry import build

    return build(kind, model_id, device=args.device, allow_cloud=args.allow_cloud)


def _bench() -> dict[str, Any]:
    from voice_lab.config import load_benchmark_config

    return load_benchmark_config()


def _profile_steps(name: str) -> list[dict[str, Any]]:
    from voice_lab.config import ConfigError

    profiles = _bench().get("preprocess_profiles", {})
    if name not in profiles:
        raise ConfigError(
            f"Unknown preprocessing profile {name!r}; defined in configs/benchmark.yaml: {sorted(profiles)}"
        )
    return list(profiles[name])


# --- commands -----------------------------------------------------------------------------------


def cmd_system_info(args: argparse.Namespace) -> int:
    from voice_lab.config import OPTIONAL_ENV, missing_credentials
    from voice_lab.providers.base import ProviderError
    from voice_lab.utils.hardware import system_info
    from voice_lab.utils.http import get_json

    info = system_info()
    root = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        info["ollama"] = {
            "url": root,
            "version": get_json(f"{root}/api/version", timeout=3).get("version"),
            "models": [m["name"] for m in get_json(f"{root}/api/tags", timeout=3).get("models", [])],
        }
    except ProviderError:
        info["ollama"] = {"url": root, "reachable": False}
    missing = missing_credentials()
    info["optional_credentials"] = {k: ("missing" if k in missing else "set") for k in OPTIONAL_ENV}
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    gpus = ", ".join(f"{g['name']} ({g['vram_gb']} GB)" for g in info["gpus"]) or "none -> CPU-only"
    print(
        f"OS         {info['os']}\nPython     {info['python']}\nCPU        {info['cpu']} "
        f"({info['cpu_cores_physical']} cores / {info['cpu_threads']} threads)\n"
        f"RAM        {info['ram_total_gb']} GB total, {info['ram_available_gb']} GB available\n"
        f"Disk free  {info['disk_free_gb']} GB\nGPU        {gpus}\nCUDA       {info['cuda'] or '-'}"
    )
    print("Packages   " + ", ".join(f"{k} {v}" for k, v in info["packages"].items() if v))
    if not_installed := [k for k, v in info["packages"].items() if not v]:
        print(f"           not installed: {', '.join(not_installed)} (uv sync --extra local --extra cpu)")
    o = info["ollama"]
    print(
        f"Ollama     {o['url']} "
        + (f"v{o['version']}, models: {', '.join(o['models']) or 'none'}" if o.get("version") else "not reachable")
    )
    print("Optional credentials (none needed for self-hosted experiments):")
    for key, why in OPTIONAL_ENV.items():
        print(f"  {key:<16} {'set' if key not in missing else 'missing':<8} {why}")
    if not info["gpus"]:
        print(
            "\nNote: CPU-only. Small/quantised models run; "
            "Whisper large, Indic Parler-TTS, >8B LLMs will be slow or infeasible."
        )
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from voice_lab.config import KINDS, load_models
    from voice_lab.download import local_status

    header = ["id", "provider", "enabled", "cloud", "commercial use", "size", "status"]
    for kind in [args.kind] if args.kind else KINDS:
        models = load_models(kind)
        rows = [
            [
                m.id,
                m.provider,
                "on" if m.enabled else "off",
                "yes" if m.external else "",
                m.commercial_use,
                m.size or "",
                local_status(m),
            ]
            for m in models.values()
        ]
        if args.json:
            print(json.dumps({kind: [dict(zip(header, r, strict=True)) for r in rows]}, ensure_ascii=False))
            continue
        print(f"\n{kind.upper()} ({len(rows)})")
        widths = [min(60, max(len(str(r[i])) for r in [header, *rows])) for i in range(len(header))]
        for r in [header, *rows]:
            print("  " + "  ".join(str(v)[:60].ljust(w) for v, w in zip(r, widths, strict=True)))
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from voice_lab.download import download

    download(_find_model(args.model), yes=args.yes)
    return 0


def cmd_asr_transcribe(args: argparse.Namespace) -> int:
    from voice_lab.audio.preprocess import load_and_prepare

    steps = _profile_steps(args.profile)
    provider = _build("asr", args.model, args)
    from voice_lab.audio.io import peak_dbfs

    audio = load_and_prepare(args.audio, steps)
    if peak_dbfs(audio) < -60:
        print(
            f"WARNING: {args.audio} is (nearly) silent - check the microphone before trusting the transcript.",
            file=sys.stderr,
        )
    result = provider.transcribe(audio)
    if args.json:
        print(
            json.dumps(
                {**result.__dict__, "load_seconds": provider.load_seconds, "preprocessing": steps},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(result.text)
        print(
            f"[{result.model}: {result.latency_seconds:.2f}s for {result.audio_duration_seconds:.2f}s audio, "
            f"RTF {result.real_time_factor:.2f}, load {provider.load_seconds:.1f}s, profile {args.profile}]",
            file=sys.stderr,
        )
    return 0


def _print_reply(result: Any, seconds: float, ttft: float | None, trace: list[dict[str, Any]]) -> None:
    for call in trace:
        print(
            f"  [tool] {call['name']}({json.dumps(call['arguments'], ensure_ascii=False)}) "
            f"-> {json.dumps(call['result'], ensure_ascii=False)}"
        )
    print(result.text)
    ttft_s = f"{ttft:.2f}s" if ttft is not None else "-"
    print(f"[{result.model}: TTFT {ttft_s}, total {seconds:.2f}s, out tokens {result.output_tokens}]", file=sys.stderr)


def cmd_llm_chat(args: argparse.Namespace) -> int:
    from voice_lab.llm.tools import run_with_tools

    provider = _build("llm", args.model, args)
    system = args.system if args.system is not None else _bench().get("llm", {}).get("system_prompt", "")
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []

    def turn(text: str) -> None:
        messages.append({"role": "user", "content": text})
        if args.tools:
            result, trace, ttft, seconds = run_with_tools(provider, messages)
        else:
            result = provider.generate(messages)
            trace, ttft, seconds = [], result.time_to_first_token, result.latency_seconds
        messages.append({"role": "assistant", "content": result.text})
        _print_reply(result, seconds, ttft, trace)

    if args.text:
        turn(args.text)
        return 0
    print("Interactive chat. Ctrl-D or 'exit' to quit.", file=sys.stderr)
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            return 0
        if text in ("exit", "quit"):
            return 0
        if text:
            turn(text)


def cmd_llm_health(args: argparse.Namespace) -> int:
    info = _build("llm", args.model, args).health()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0 if info.get("ok") else 1


def cmd_tts_synthesize(args: argparse.Namespace) -> int:
    from voice_lab import config

    provider = _build("tts", args.model, args)
    out = (
        Path(args.out)
        if args.out
        else config.RESULTS_DIR / "tts" / "adhoc" / f"{args.model}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.wav"
    )
    r = provider.synthesize(args.text, out, voice=args.voice)
    print(r.audio_path)
    print(
        f"[{r.model}: {r.latency_seconds:.2f}s for {r.audio_duration_seconds:.2f}s audio @ {r.sample_rate} Hz, "
        f"RTF {r.real_time_factor:.2f}, load {provider.load_seconds:.1f}s; original output: {r.original_path}]",
        file=sys.stderr,
    )
    return 0


def cmd_vad_detect(args: argparse.Namespace) -> int:
    from voice_lab.audio.io import load_audio

    result = _build("vad", args.model, args).detect_speech(load_audio(args.audio))
    for start, end in result.segments:
        print(f"{start:8.2f} - {end:8.2f} s")
    print(
        f"[{len(result.segments)} segments, {result.speech_seconds:.2f}s speech / "
        f"{result.audio_duration_seconds:.2f}s, "
        f"{result.latency_seconds * 1000:.0f} ms]",
        file=sys.stderr,
    )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from voice_lab import config
    from voice_lab.benchmarking import runner
    from voice_lab.evaluation.manifest import load_manifest, manifest_sha
    from voice_lab.providers.registry import select

    bench = _bench()
    kind = args.kind
    if kind == "pipeline":
        return _benchmark_pipeline(args, bench)
    manifest_kind = "asr" if kind == "telephone" else kind
    dataset = Path(args.dataset or bench["datasets"][manifest_kind])
    rows = load_manifest(dataset, manifest_kind)[: args.limit]
    run_config: dict[str, Any] = {"benchmark": kind, "limit": args.limit}
    output = Path(args.output) if args.output else None
    if kind in ("asr", "telephone"):
        names = args.profile or (
            bench["telephone_suite"] if kind == "telephone" else bench.get("asr_profiles", ["original"])
        )
        profiles = {n: _profile_steps(n) for n in names}
        jobs = runner.asr_jobs(select("asr", args.model), profiles)
        evaluate: runner.Evaluate = runner.eval_asr
        run_config["profiles"] = profiles
        output = output or (config.RESULTS_DIR / "telephone" if kind == "telephone" else None)
    elif kind == "llm":
        jobs = [runner.Job(s, "default", bench.get("llm", {})) for s in select("llm", args.model)]
        evaluate = runner.eval_llm
    else:
        judge = None
        if judge_id := args.asr_judge or bench.get("tts", {}).get("asr_judge"):
            judge = _build("asr", judge_id, args)
            judge.load()
        jobs = [
            runner.Job(s, v, {"asr_judge": judge_id})
            for s in select("tts", args.model)
            for v in (args.voice or ["default"])
        ]
        evaluate = runner.make_eval_tts(judge)
    print(f"Benchmark {kind}: {len(jobs)} job(s) x {len(rows)} samples from {dataset}")
    run_dir = runner.run_benchmark(
        "asr" if kind == "telephone" else kind,
        rows,
        jobs,
        evaluate,
        dataset=dataset,
        dataset_sha=manifest_sha(dataset),
        output=output,
        workers=args.workers,
        force=args.force,
        device=args.device,
        allow_cloud=args.allow_cloud,
        run_config=run_config,
    )
    print(f"Results: {run_dir}\n  summary: {run_dir / 'summary.csv'}\n  database: {config.RESULTS_DIR / 'lab.db'}")
    print("Next: voice-lab reports")
    return 0


def _benchmark_pipeline(args: argparse.Namespace, bench: dict[str, Any]) -> int:
    from voice_lab import config
    from voice_lab.evaluation.manifest import load_manifest, resolve_audio
    from voice_lab.pipeline import converse, summarize_timings

    cfg = bench.get("pipeline", {})
    dataset = Path(args.dataset or bench["datasets"]["asr"])
    rows = load_manifest(dataset, "pipeline")[: args.limit]
    ids = {k: getattr(args, k, None) or cfg.get(k) for k in ("vad", "asr", "llm", "tts")}
    vad = _build("vad", ids["vad"], args) if ids["vad"] else None
    asr, llm, tts = (_build(k, ids[k], args) for k in ("asr", "llm", "tts"))
    run_dir = Path(args.output or config.RESULTS_DIR / "system") / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-pipeline"
    results = []
    for row in rows:
        r = converse(
            resolve_audio(row["audio"]),
            asr=asr,
            llm=llm,
            tts=tts,
            vad=vad,
            out_dir=run_dir / row["id"],
            system_prompt=bench.get("llm", {}).get("system_prompt", ""),
            tools=args.tools,
        )
        results.append({"id": row["id"], **r})
        print(f"  {row['id']}: total {r['timings'].get('total', float('nan')):.2f}s {r.get('error', '')}")
    summary = {"models": ids, "dataset": str(dataset), "summary": summarize_timings(results), "results": results}
    (run_dir / "pipeline.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["summary"], indent=2))
    print(f"Results: {run_dir / 'pipeline.json'}")
    return 0


def cmd_conversation(args: argparse.Namespace) -> int:
    from voice_lab import config
    from voice_lab.pipeline import converse, waterfall

    bench = _bench()
    cfg = bench.get("pipeline", {})
    ids = {k: getattr(args, k) or cfg.get(k) for k in ("vad", "asr", "llm", "tts")}
    vad = _build("vad", ids["vad"], args) if ids["vad"] and not args.no_vad else None
    out = (
        Path(args.out)
        if args.out
        else config.RESULTS_DIR / "system" / f"conversation-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    result = converse(
        args.audio,
        asr=_build("asr", ids["asr"], args),
        llm=_build("llm", ids["llm"], args),
        tts=_build("tts", ids["tts"], args),
        vad=vad,
        out_dir=out,
        system_prompt=bench.get("llm", {}).get("system_prompt", ""),
        tools=args.tools,
    )
    print(f"Transcript: {result.get('transcript', '-')}\nResponse:   {result.get('response', '-')}")
    if result.get("error"):
        print(f"Error:      {result['error']}")
    print(f"Audio:      {result.get('response_audio', '-')}\n\n{waterfall(result['timings'])}")
    print(f"\n(model load times, excluded above: {result['load_seconds']})\nDetails: {out / 'result.json'}")
    return 0 if not result.get("error") else 1


def cmd_ui(args: argparse.Namespace) -> int:
    from voice_lab.providers.base import ProviderError

    try:
        from voice_lab.ui import launch
    except ImportError as e:
        raise ProviderError(f"The UI needs Gradio ({e}). Install: uv sync --extra local --extra cpu --extra ui") from e
    if args.host not in ("127.0.0.1", "localhost"):
        print(f"WARNING: binding to {args.host} makes the lab reachable from your network.", file=sys.stderr)
    launch(args.host, args.port, args.open)
    return 0


def cmd_reports(args: argparse.Namespace) -> int:
    from voice_lab.benchmarking.report import generate

    for path in generate(Path(args.out) if args.out else None):
        print(path)
    return 0


def _latest_tts_run(run: str | None) -> Path:
    from voice_lab.benchmarking.report import latest_run
    from voice_lab.config import ConfigError

    run_dir = Path(run) if run else latest_run("tts")
    if run_dir is None or not run_dir.is_dir():
        raise ConfigError("No TTS run found. Run `voice-lab benchmark tts` first or pass --run.")
    return run_dir


def cmd_human_sheet(args: argparse.Namespace) -> int:
    from voice_lab.evaluation.human import make_sheet

    out = make_sheet(_latest_tts_run(args.run), seed=args.seed)
    print(f"Rater pack: {out}/clips + {out}/ratings_template.csv (1-5 per criterion).")
    print(f"Keep {out}/key.json away from raters. Import with: voice-lab human-eval import --ratings FILE --rater R01")
    return 0


def cmd_human_import(args: argparse.Namespace) -> int:
    from voice_lab import config
    from voice_lab.evaluation.human import import_ratings

    summary = import_ratings(_latest_tts_run(args.run), Path(args.ratings), args.rater, config.RESULTS_DIR / "lab.db")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_dataset_validate(args: argparse.Namespace) -> int:
    from collections import Counter

    from voice_lab.evaluation.manifest import load_manifest

    rows = load_manifest(args.manifest, args.kind, check_files=not args.no_files)
    print(f"OK: {len(rows)} rows")
    for category, n in sorted(Counter(r.get("category", "-") for r in rows).items()):
        print(f"  {category:<24} {n}")
    return 0


def cmd_dataset_synth(args: argparse.Namespace) -> int:
    """TTS-generated ASR test audio. Labelled `synthetic` - never a substitute for real recorded speech."""
    from voice_lab import config
    from voice_lab.evaluation.manifest import load_manifest

    rows = load_manifest(args.prompts, "tts")[: args.limit]
    provider = _build("tts", args.tts, args)
    audio_dir = config.ROOT / "data" / "synthetic" / args.tts
    manifest = config.ROOT / "data" / "manifests" / f"asr_synthetic_{args.tts}.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            r = provider.synthesize(row["text"], audio_dir / row["id"], voice=args.voice)
            record = {
                "id": row["id"],
                "audio": str(Path(r.audio_path).relative_to(config.ROOT)),
                "language": row.get("language", "ne"),
                "text": row["text"],
                "category": row.get("category"),
                "entities": row.get("entities", {}),
                "speaker_id": f"{args.tts}:{args.voice or 'default'}",
                "sample_rate": r.sample_rate,
                "source": f"synthetic-tts:{args.tts}",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  {row['id']}: {r.audio_duration_seconds:.1f}s")
    print(
        f"Wrote {manifest} ({len(rows)} rows). SYNTHETIC audio: use for pipeline smoke tests, not for model selection."
    )
    return 0


# --- parser -------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default=None, help="cpu | cuda | auto (default: model's params.device, else cpu)")
    common.add_argument(
        "--allow-cloud", action="store_true", help="confirm sending data to models marked external (e.g. Gemini)"
    )

    p = argparse.ArgumentParser(
        prog="voice-lab", description="Nepali voice-agent research lab (isolated, local-first)."
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("system-info", help="hardware, packages, Ollama, optional credentials")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_system_info)

    s = sub.add_parser("models", help="list registered models with licence and local status")
    s.add_argument("--kind", choices=["asr", "llm", "tts", "vad"])
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("download", help="explicitly download a model (shows size + licence, asks first)")
    s.add_argument("model")
    s.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    s.set_defaults(func=cmd_download)

    asr = sub.add_parser("asr").add_subparsers(dest="action", required=True)
    s = asr.add_parser("transcribe", parents=[common])
    s.add_argument("audio")
    s.add_argument("--model", required=True)
    s.add_argument("--profile", default="original", help="preprocessing profile from configs/benchmark.yaml")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_asr_transcribe)

    llm = sub.add_parser("llm").add_subparsers(dest="action", required=True)
    s = llm.add_parser("chat", parents=[common])
    s.add_argument("--model", required=True)
    s.add_argument("--text", help="single message (omit for interactive chat)")
    s.add_argument("--system", help="override the system prompt from configs/benchmark.yaml")
    s.add_argument("--tools", action="store_true", help="enable the FAKE local test tools")
    s.set_defaults(func=cmd_llm_chat)
    s = llm.add_parser("health", parents=[common])
    s.add_argument("--model", required=True)
    s.set_defaults(func=cmd_llm_health)

    tts = sub.add_parser("tts").add_subparsers(dest="action", required=True)
    s = tts.add_parser("synthesize", parents=[common])
    s.add_argument("--model", required=True)
    s.add_argument("--text", required=True)
    s.add_argument("--voice")
    s.add_argument("--out", help="output .wav path")
    s.set_defaults(func=cmd_tts_synthesize)

    vad = sub.add_parser("vad").add_subparsers(dest="action", required=True)
    s = vad.add_parser("detect", parents=[common])
    s.add_argument("audio")
    s.add_argument("--model", default="silero-vad")
    s.set_defaults(func=cmd_vad_detect)

    s = sub.add_parser("benchmark", parents=[common], help="run a benchmark; results -> results/ + results/lab.db")
    s.add_argument("kind", choices=["asr", "llm", "tts", "telephone", "pipeline"])
    s.add_argument("--model", action="append", help="model id (repeatable); default: all enabled models")
    s.add_argument("--dataset", help="manifest path (default from configs/benchmark.yaml)")
    s.add_argument("--limit", type=int, help="first N samples only")
    s.add_argument("--output", help="results directory (default results/<kind>)")
    s.add_argument("--workers", type=int, default=1, help="parallel samples per model (keep 1 for local models)")
    s.add_argument("--force", action="store_true", help="re-run samples that already have cached results")
    s.add_argument("--profile", action="append", help="ASR preprocessing profile (repeatable)")
    s.add_argument("--voice", action="append", help="TTS voice/speaker (repeatable)")
    s.add_argument("--asr-judge", help="TTS: ASR model used as intelligibility proxy")
    for k in ("vad", "asr", "llm", "tts"):
        s.add_argument(f"--{k}-model", dest=k, help=f"pipeline: {k} model")
    s.add_argument("--tools", action="store_true", help="pipeline: enable fake tools")
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("conversation", parents=[common], help="recorded audio -> VAD -> ASR -> LLM -> TTS (local)")
    s.add_argument("--audio", required=True)
    for k in ("vad", "asr", "llm", "tts"):
        s.add_argument(f"--{k}", help=f"{k} model id (default from configs/benchmark.yaml pipeline:)")
    s.add_argument("--no-vad", action="store_true")
    s.add_argument("--tools", action="store_true", help="enable the FAKE local test tools")
    s.add_argument("--out")
    s.set_defaults(func=cmd_conversation)

    s = sub.add_parser("ui", help="local web UI (Gradio) for every feature, on http://127.0.0.1:7860")
    s.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 unless you mean to expose it")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("--open", action="store_true", help="open a browser tab")
    s.set_defaults(func=cmd_ui)

    s = sub.add_parser("reports", help="write markdown reports to reports/")
    s.add_argument("--out")
    s.set_defaults(func=cmd_reports)

    he = sub.add_parser("human-eval", help="blind 1-5 human rating of TTS output").add_subparsers(
        dest="action", required=True
    )
    s = he.add_parser("sheet", help="anonymised clips + rating CSV for the latest (or given) TTS run")
    s.add_argument("--run", help="results/tts/<run_id> (default: latest)")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_human_sheet)
    s = he.add_parser("import", help="store one rater's filled CSV (anonymous rater id)")
    s.add_argument("--run")
    s.add_argument("--ratings", required=True)
    s.add_argument("--rater", required=True, help="pseudonym such as R01 - never a real name")
    s.set_defaults(func=cmd_human_import)

    ds = sub.add_parser("dataset").add_subparsers(dest="action", required=True)
    s = ds.add_parser("validate")
    s.add_argument("manifest")
    s.add_argument("--kind", required=True, choices=["asr", "llm", "tts", "pipeline"])
    s.add_argument("--no-files", action="store_true", help="skip audio-file existence check")
    s.set_defaults(func=cmd_dataset_validate)
    s = ds.add_parser("synth", parents=[common], help="make SYNTHETIC ASR test audio with a TTS model")
    s.add_argument("--tts", required=True)
    s.add_argument("--voice")
    s.add_argument("--prompts", default="data/manifests/asr_prompts.jsonl")
    s.add_argument("--limit", type=int)
    s.set_defaults(func=cmd_dataset_synth)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "download":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never fetch weights implicitly (must precede HF imports)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    from voice_lab.config import ConfigError, load_dotenv
    from voice_lab.evaluation.manifest import ManifestError
    from voice_lab.providers.base import ProviderError

    load_dotenv()
    if not hasattr(args, "device"):
        args.device, args.allow_cloud = None, False
    try:
        return int(args.func(args) or 0)
    except (ConfigError, ProviderError, ManifestError, ValueError, FileNotFoundError) as e:
        if args.verbose:
            raise
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
