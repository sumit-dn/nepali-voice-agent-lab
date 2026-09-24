# TTS comparison

> Evidence only. This report does not rank models or pick a winner. Weigh quality, latency, hardware, concurrency and licensing for the actual use case. Numbers come only from runs listed below.

- Run: `/home/digital-nepal/Documents/dn-ai/nepali-voice-agent-lab/results/tts/20260924T064845937Z-tts`
- Dataset: `data/manifests/tts.jsonl` (sha 679579d89cff)
- Hardware: Intel(R) Core(TM) i5-7400 CPU @ 3.00GHz (4 threads), 16.1 GB RAM, GPU: none (CPU-only)

| Model | Voice | N | Failed | Latency p50 (s) | RTF | Sample rate | ASR-judge CER | Peak RAM (MB) | Load (s) | License | Commercial | Error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-ne-google-x-low | default | 12 | 0 | 0.220 | 0.090 | 16000 | 0.261 | – | – | CC-BY-SA-4.0 (voice data); engine GPL-3.0-or-later | REVIEW REQUIRED | – |
| xtts-ne-oshara-e20 | default | 12 | 0 | 16.953 | 3.766 | 24000 | 0.253 | 2968 | 172.557 | Coqui Public Model License 1.0.0 (inherited from coqui/XTTS-v2, per the fine-tune's own card) | no | – |

ASR-judge CER is an intelligibility *proxy* (an ASR model transcribes the TTS output); ASR errors confound it. It is not MOS. Naturalness/pronunciation/prosody need the human rating protocol in docs/evaluation.md.
