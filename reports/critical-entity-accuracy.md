# Critical-entity accuracy (numbers, names, dates, ...)

> Evidence only. This report does not rank models or pick a winner. Weigh quality, latency, hardware, concurrency and licensing for the actual use case. Numbers come only from runs listed below.

- Run: `/home/digital-nepal/Documents/dn-ai/nepali-voice-agent-lab/results/asr/20260923T102131574Z-asr`
- Dataset: `data/manifests/asr_synthetic_piper-ne-google-x-low.jsonl` (sha c6f0b7694bfa)
- Hardware: Intel(R) Core(TM) i5-7400 CPU @ 3.00GHz (4 threads), 16.1 GB RAM, GPU: none (CPU-only)

| Model | Preprocessing | WER | account_numbers | addresses | currency | dates | mixed_language | names | numbers | otps | phone_numbers | times |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| indicconformer-600m-rnnt | original | 0.463 | 0.00 (0/1) | 0.00 (0/3) | 0.00 (0/2) | 0.00 (0/1) | 0.00 (0/22) | 0.00 (0/4) | 0.00 (0/1) | 0.00 (0/1) | 0.00 (0/1) | 1.00 (1/1) |
| indicconformer-600m | original | 0.476 | 0.00 (0/1) | 0.33 (1/3) | 0.00 (0/2) | 0.00 (0/1) | 0.00 (0/22) | 0.00 (0/4) | 0.00 (0/1) | 0.00 (0/1) | 0.00 (0/1) | 1.00 (1/1) |

An entity counts as correct only if it appears in the transcript after normalisation; numeric entities are compared as digit strings (Devanagari/ASCII digits and grouping ignored). Numbers spoken as words and transcribed as words will count as misses - see docs/evaluation.md.
