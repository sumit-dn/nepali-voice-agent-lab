# Nepali Voice-Agent Lab

A standalone, local-first research lab for measuring whether open-weight ASR, LLM and TTS models can
replace paid cloud AI in a Nepali / Nepali–English phone assistant. It produces evidence (quality,
latency, hardware, licensing), not a pre-picked winner.

> **Isolation.** This repo shares nothing with the deployed call-assistant or VoiceStudio: no code,
> config, `.env`, keys, database, PBX or media endpoints. Every tool is a fake local stub, the knowledge
> base is fictional, and no real calls are involved.

## 1. Goal
Build a reproducible way to compare **quality, latency, resource use, concurrency, and licensing** for
self-hosted models, with optional cloud baselines, and then assemble a self-hosted voice-agent prototype.

## 2. Why
Cloud ASR, LLM and TTS cost money per minute and send customer audio off-site. Whether open models are
good enough in **Nepali and code-switched Nepali–English, over telephone audio, with numbers and names
intact**, is an empirical question. This lab answers it with measurements.

## 3. Architecture
```
configs/*.yaml ─► registry ─► Provider (ASR | LLM | TTS | VAD) ─► runner ─► results/ (JSON, CSV, SQLite) ─► reports/
                                  ▲
       adapters: hf-asr, indicconformer, openai-compat (Ollama/vLLM/Gemini), piper, parler, silero, gemini-*
```
Benchmark code talks only to the four provider interfaces (`src/voice_lab/providers/base.py`), so models
are swapped by editing YAML, never Python. See [docs/architecture.md](docs/architecture.md).

## 4. Installation
```bash
make install              # uv sync: Python 3.11, CPU torch, transformers, piper-tts, silero-vad, dev tools
make install-gpu          # same, but CUDA 12.8 torch
cp .env.example .env      # optional - nothing in it is required
.venv/bin/voice-lab system-info
```
Requires [uv](https://docs.astral.sh/uv/). LLMs are served by [Ollama](https://ollama.com) or vLLM, not inside this process.

### Web UI
```bash
make ui                   # = voice-lab ui --open  ->  http://127.0.0.1:7860
```
The UI has one tab per feature:
- System & models, including downloads with a size and licence check
- Speech → Text, with VAD
- Text → Speech
- Chat, with the fake tools
- Conversation: speak into your microphone and get a spoken reply
- Benchmarks
- Reports
- Human rating: blind 1–5 scores stored per anonymous rater

The UI binds to 127.0.0.1 only, with no share link and Gradio analytics off. It uses the same providers as the CLI, and its
benchmarks and downloads run the real CLI underneath. Loaded models stay in memory; use **Unload models** to free RAM.

## 5. GPU requirements
Documented requirements, taken from the model cards, are in each registry entry's `hardware` field.
Roughly:
- Whisper large-v3-turbo needs ~6 GB VRAM.
- Indic Parler-TTS needs a CUDA GPU for usable speed.
- Qwen3-14B Q4 needs ~11–12 GB.
- Mistral Small 24B Q4 needs a ~24 GB GPU.

These figures come from the model cards; they are not measured here.

## 6. CPU-only limitations
Numbers measured on the development box (i5-7400, 4 threads, 16 GB RAM, no GPU):
- IndicConformer-600M CTC: RTF ≈ 0.19 and ≈ 3 GB RAM.
- Piper TTS: RTF ≈ 0.08–0.11.
- Llama-3.1-8B Q4 via Ollama: **33–40 s time to first token**, which is unusable for live calls.

Conclusion: a CPU box can do ASR, TTS and VAD experiments, but LLM latency needs a GPU.
See [docs/hardware.md](docs/hardware.md).

## 7. Model installation
Nothing downloads implicitly: the CLI runs Hugging Face in offline mode. `voice-lab download <id>` is the
only command that fetches anything. It shows the exact size, licence, commercial status and gating first,
and then asks for confirmation.
```bash
voice-lab models                           # every candidate: provider, on/off, commercial use, size, local status
voice-lab download piper-ne-google-x-low   # 28 MB
voice-lab download qwen3-8b                # runs `ollama pull qwen3:8b` (5.2 GB) after confirmation
```
Gated checkpoints (IndicConformer, Indic Parler-TTS, Gemma 3, Llama) need you to accept their terms on
Hugging Face and set `HF_TOKEN`.

## 8. Dataset format
Datasets are JSONL manifests; see [data/README.md](data/README.md).
- `data/manifests/llm.jsonl`: 22 cases with deterministic validators and fake tools.
- `tts.jsonl`: 30 texts across 15 categories.
- `asr_prompts.jsonl`: a reading script with entity annotations.
- The real ASR set, `data/manifests/asr.jsonl`, must come from **real recordings** with consent. The
  repo ships no fabricated ground truth.

## 9. ASR testing
```bash
voice-lab asr transcribe clip.wav --model indicconformer-600m [--profile telephone_8k]
voice-lab benchmark asr        [--model ID ...] [--profile clean_16k --profile clean_8k] [--limit N]
voice-lab benchmark telephone  # clean 16k / clean 8k / 8k+G.711 / noisy 8k+G.711
```
Metrics: WER, CER, per-category WER, and critical-entity accuracy (phone, OTP, account numbers, currency,
dates, names, addresses), plus mixed-language word recall, an error-tag analysis, latency, RTF, and
RAM/VRAM use. See [docs/asr.md](docs/asr.md) and [docs/evaluation.md](docs/evaluation.md).

## 10. LLM testing
```bash
voice-lab llm health --model qwen3-8b
voice-lab llm chat --model qwen3-8b --text "मेरो balance कति छ? Customer ID C-1001" --tools
voice-lab benchmark llm
```
The same model can run via Ollama or vLLM through one OpenAI-compatible client that streams every call,
so time-to-first-token (TTFT) is measured the same way on each. See [docs/llm.md](docs/llm.md).

## 11. TTS testing
```bash
voice-lab tts synthesize --model piper-ne-google-x-low --text "तपाईंको appointment tomorrow बिहान १० बजे छ।"
voice-lab benchmark tts --asr-judge indicconformer-600m
```
Each synthesis saves the raw model output (`*.orig.wav`) and a normalised copy (mono, 16-bit PCM WAV).
The ASR-judge CER is an intelligibility *proxy*, not MOS; the human rating protocol is in
[docs/evaluation.md](docs/evaluation.md).

## 12. Benchmarking
Each `benchmark` run writes `results/<kind>/<run_id>/` with one JSON per model and variant (config,
licence, hardware, per-sample outputs), plus `summary.csv` and `samples.csv`. It also indexes everything in
`results/lab.db` (SQLite).

Samples whose model config, dataset hash and sample are unchanged are **reused, not re-run**; pass
`--force` to re-run them. `--workers N` parallelises samples; keep it at 1 for in-process models.

The recorded-audio pipeline:
```bash
voice-lab conversation --audio clip.wav --asr indicconformer-600m --llm qwen3-8b --tts piper-ne-google-x-low
```
It prints the transcript, the response, an output WAV and a timing waterfall (VAD/ASR/LLM/TTS, ASR RTF,
LLM TTFT, TTS RTF). `voice-lab benchmark pipeline` runs the same thing over a manifest.

## 13. Results
`voice-lab reports` writes `reports/*.md`:
- ASR comparison, ASR error analysis, critical-entity accuracy, telephone comparison
- LLM comparison, TTS comparison
- licensing, hardware profile

The reports show evidence and trade-offs and never pick a winner. Results produced from synthetic
(TTS-generated) audio are pipeline smoke tests only.

## 14. Licensing
Every registry entry records the licence of **that exact checkpoint**, copied from its own card with a
verification date. Anything unclear is marked `REVIEW REQUIRED`. `reports/licensing.md` is generated from
the registries; [docs/licensing.md](docs/licensing.md) summarises the open questions. "Open-source",
"open-weight", "free" and "commercially usable" mean different things.

## 15. Privacy
Everything stays local by default:
- Cloud models are marked `external: true`, are disabled by default, and refuse to run without
  `--allow-cloud`. With it, they print `WARNING: This experiment sends audio to Google Gemini API`.
- Free-tier Gemini content may be used by Google and reviewed by humans, so use only a personal or test
  key and never real customer data.
- Audio, results and `.env` are git-ignored.
- ASR manifests carry `consent` and `deidentified` fields.

## 16. Future telephony integration (not implemented)
Integration comes last, after streaming, barge-in, concurrency and GPU sizing have evidence behind them.
The expected path: a separate gateway service in front of these same provider interfaces, receiving 8 kHz
μ-law RTP/WebSocket audio. Before that exists it will need load tests against a **non-production** PBX.
See [docs/deployment.md](docs/deployment.md).

## Development
```bash
make check          # ruff + mypy + pytest (fake providers, no downloads, no network)
make test-integration
```
