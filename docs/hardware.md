# Hardware

`voice-lab system-info` prints CPU, RAM, disk, GPU/VRAM, CUDA, and the torch and runtime versions. Each
benchmark stores the same data in its results, and `reports/hardware-profile.md` tabulates load time and
peak RAM/VRAM per model.

## Development box (measured 2026-09-23, synthetic audio)
Intel i5-7400 (4 cores / 4 threads), 16 GB RAM, **no GPU**, torch 2.14 CPU.

| component | load | speed | peak process RAM |
|---|---|---|---|
| IndicConformer-600M, CTC | ~8 s | RTF 0.19 | ~3.1 GB |
| IndicConformer-600M, RNNT | ~7 s | RTF 0.35 | ~5.3 GB |
| Piper ne x_low / medium | ≤1.5 s | RTF 0.08–0.11 | small (the table figure includes the ASR judge) |
| Silero VAD | <1 s | ~0.1 s for 4 s audio | small |
| Llama 3.1 8B Q4 (Ollama) | – | TTFT 33–40 s | ~5 GB in the server |

System RAM peaked around 13 GB of 16 GB while the ASR benchmark ran next to the desktop session. Running
IndicConformer and an 8B LLM on this box at the same time is at the limit.

## Sizing guidance (from model cards, not measured here)
- **ASR:** Whisper large-v3-turbo needs ~6 GB VRAM. IndicConformer is feasible on CPU at RTF < 1, but
  concurrent calls multiply RAM and CPU.
- **LLM:**
  - Qwen3-8B Q4: ~6–8 GB. Qwen3-14B Q4: ~11–12 GB.
  - BF16 needs ~2× the parameter count in GB.
  - For sub-second TTFT a GPU is required; the target test is a 24 GB card running vLLM.
- **TTS:** Piper runs on CPU. Indic Parler-TTS (0.9B, autoregressive) needs a GPU for real-time speech.

Concurrency testing (1/2/5/10/20 simultaneous calls) is a remaining task, and a single fast request
proves nothing about capacity.
