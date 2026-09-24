"""Recorded-audio conversation: VAD -> ASR -> LLM -> TTS with a timing waterfall.

This is NOT a telephony integration: input is a file, output is a file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from voice_lab.audio.io import Audio, load_audio
from voice_lab.llm.tools import run_with_tools
from voice_lab.providers.base import ASRProvider, LLMProvider, TTSProvider, VADProvider

STAGES = ("vad", "asr", "llm", "tts")


def crop_to_speech(audio: Audio, segments: list[tuple[float, float]], pad: float = 0.2) -> Audio:
    start = max(0, int((segments[0][0] - pad) * audio.sample_rate))
    end = min(audio.samples.shape[0], int((segments[-1][1] + pad) * audio.sample_rate))
    return Audio(audio.samples[start:end], audio.sample_rate)


def converse(
    audio_path: str | Path,
    *,
    asr: ASRProvider,
    llm: LLMProvider,
    tts: TTSProvider,
    vad: VADProvider | None,
    out_dir: Path,
    system_prompt: str,
    tools: bool = False,
) -> dict[str, Any]:
    audio = load_audio(audio_path)
    providers = [p for p in (vad, asr, llm, tts) if p is not None]
    for p in providers:
        p.load()  # load time is reported separately; it is not part of response latency
    timings: dict[str, Any] = {"audio_duration": round(audio.duration, 3)}
    result: dict[str, Any] = {
        "audio": str(audio_path),
        "models": {k: p.spec.id for k, p in zip(STAGES, (vad, asr, llm, tts), strict=True) if p is not None},
        "load_seconds": {p.spec.id: round(p.load_seconds or 0, 3) for p in providers},
        "timings": timings,
    }
    speech = audio
    if vad is not None:
        v = vad.detect_speech(audio)
        timings["vad"] = v.latency_seconds
        result["speech_segments"] = v.segments
        if not v.segments:
            result["error"] = "VAD found no speech"
            return _finish(result, out_dir)
        speech = crop_to_speech(audio, v.segments)
    heard = asr.transcribe(speech)
    timings["asr"], timings["asr_rtf"] = heard.latency_seconds, heard.real_time_factor
    result["transcript"] = heard.text
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": heard.text}]
    if tools:
        reply, trace, ttft, llm_seconds = run_with_tools(llm, messages)
        result["tool_calls"] = trace
    else:
        reply = llm.generate(messages)
        ttft, llm_seconds = reply.time_to_first_token, reply.latency_seconds
    timings["llm"], timings["llm_ttft"] = llm_seconds, ttft
    result["response"] = reply.text
    if not reply.text.strip():
        result["error"] = "LLM returned no text to speak"
        return _finish(result, out_dir)
    spoken = tts.synthesize(reply.text, out_dir / "response.wav")
    timings["tts"], timings["tts_rtf"] = spoken.latency_seconds, spoken.real_time_factor
    result["response_audio"] = spoken.audio_path
    timings["total"] = sum(timings.get(s) or 0.0 for s in STAGES)
    return _finish(result, out_dir)


def _finish(result: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def waterfall(timings: dict[str, Any]) -> str:
    lines = [f"{s.upper():<9} {timings[s]:6.2f} s" for s in STAGES if timings.get(s) is not None]
    extras = [
        f"ASR RTF {timings['asr_rtf']:.2f}" if timings.get("asr_rtf") is not None else "",
        f"LLM TTFT {timings['llm_ttft']:.2f} s" if timings.get("llm_ttft") is not None else "",
        f"TTS RTF {timings['tts_rtf']:.2f}" if timings.get("tts_rtf") is not None else "",
    ]
    total = f"{'Total':<9} {timings['total']:6.2f} s" if "total" in timings else "Total     (incomplete)"
    return "\n".join([*lines, "-" * 18, total, "  ".join(e for e in extras if e)])


def summarize_timings(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(results), "failed": sum("error" in r for r in results)}
    for key in (*STAGES, "llm_ttft", "total"):
        values = [r["timings"][key] for r in results if r["timings"].get(key) is not None]
        if values:
            out[f"{key}_p50"] = round(float(np.percentile(values, 50)), 3)
            out[f"{key}_p95"] = round(float(np.percentile(values, 95)), 3)
    return out
