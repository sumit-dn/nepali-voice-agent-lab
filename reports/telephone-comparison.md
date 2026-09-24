# Telephone-audio ASR comparison

> Evidence only. This report does not rank models or pick a winner. Weigh quality, latency, hardware, concurrency and licensing for the actual use case. Numbers come only from runs listed below.

- Run: `/home/digital-nepal/Documents/dn-ai/nepali-voice-agent-lab/results/telephone/20260923T113217125Z-asr`
- Dataset: `data/manifests/asr_synthetic_piper-ne-google-x-low.jsonl` (sha c6f0b7694bfa)
- Hardware: Intel(R) Core(TM) i5-7400 CPU @ 3.00GHz (4 threads), 16.1 GB RAM, GPU: none (CPU-only)

## Overall

| Model | Preprocessing | N | Failed | WER | CER | Latency p50 (s) | p95 (s) | RTF | Peak RAM (MB) | Peak VRAM (MB) | Load (s) | License | Commercial | Error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| indicconformer-600m | clean_16k | 25 | 0 | 0.476 | 0.321 | 0.506 | 1.083 | 0.207 | – | – | – | MIT | yes | – |
| indicconformer-600m | clean_8k | 25 | 0 | 0.463 | 0.308 | 0.436 | 0.648 | 0.164 | – | – | – | MIT | yes | – |
| indicconformer-600m | noisy_telephone_8k | 25 | 0 | 0.555 | 0.375 | 0.422 | 0.574 | 0.159 | – | – | – | MIT | yes | – |
| indicconformer-600m | telephone_8k | 25 | 0 | 0.469 | 0.313 | 0.442 | 0.604 | 0.159 | – | – | – | MIT | yes | – |

## WER by category

| Model | Preprocessing | account_numbers | addresses | basic_nepali | conjunct_characters | conversational | currency | dates | devanagari | english | long_nepali | names | nepali_english_mixed | numbers | otps | phone_numbers | proper_nouns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| indicconformer-600m | clean_16k | 2.000 | 0.250 | 0.100 | 0.077 | 0.364 | 0.500 | 0.538 | 0.400 | 1.000 | 0.087 | 0.182 | 0.762 | 0.600 | 0.111 | 2.200 | 0.750 |
| indicconformer-600m | clean_8k | 2.000 | 0.333 | 0.200 | 0.077 | 0.364 | 0.500 | 0.615 | 0.000 | 1.286 | 0.043 | 0.182 | 0.619 | 0.600 | 0.111 | 2.200 | 0.500 |
| indicconformer-600m | noisy_telephone_8k | 2.200 | 0.583 | 0.100 | 0.154 | 0.364 | 0.500 | 0.615 | 0.400 | 1.000 | 0.174 | 0.182 | 0.762 | 0.600 | 0.111 | 2.400 | 1.500 |
| indicconformer-600m | telephone_8k | 2.000 | 0.333 | 0.200 | 0.077 | 0.364 | 0.500 | 0.462 | 0.000 | 1.286 | 0.043 | 0.182 | 0.762 | 0.600 | 0.111 | 2.200 | 0.500 |

WER/CER are corpus-level over normalised text (NFC; drop ZWJ/ZWNJ; Devanagari digits->ASCII; drop digit-group commas; Unicode punctuation+symbols (incl. danda) -> space; lowercase; collapse whitespace).
