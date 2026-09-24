# ASR lab

Candidates live in `configs/asr.yaml`. Every entry was verified against its own model card on 2026-09-23.
The table below covers the enabled or runnable ones.

| id | runtime | licence (checkpoint) | commercial | notes |
|---|---|---|---|---|
| indicconformer-600m / -rnnt | indicconformer (ONNX, remote code) | MIT | yes | Gated (auto). Runs on CPU. Emits Devanagari only |
| whisper-tiny / small / large-v3-turbo | hf-asr | MIT (GitHub) / Apache-2.0 (HF metadata) | yes | Language forced to `nepali` |
| whisper-small-ne-dragneel | hf-asr | Apache-2.0 | REVIEW REQUIRED | Trained on OpenSLR-54 (CC BY-SA) |
| indicwav2vec-ne-sumanpaudel | hf-asr (CTC) | CC-BY-NC-4.0 | no | Research reference only |
| whisper-turbo-ne-sumanpaudel | hf-asr | CC-BY-NC-4.0 | no | Research reference only |
| gemini-3.8-flash-asr, gemini-3.5-transcribe | gemini-asr | Gemini API terms | conditional | Cloud, off by default |

Tracked entries without an adapter (`provider: none`):
- IndicConformer Nepali NeMo, which needs the AI4Bharat NeMo fork.
- MMS-1B-all `npi`, which is CC-BY-NC.
- omniASR-CTC-1B, Apache-2.0: **worth adding**.
- Qwen3-ASR, which has **no official Nepali**.
- sidskarki Qwen3-ASR-Nepali, which has a custom licence with a revenue cap.

## Findings from implementation
- **IndicConformer ignores `revision`.** The repo's `from_pretrained` calls `snapshot_download(repo)`
  without a revision. Online it would silently load whatever is on `main`; offline it fails. Our adapter
  loads the pinned code and pinned snapshot explicitly (`asr/indicconformer.py`).
- **IndicConformer's vocabulary is Devanagari-only.** English words come out transliterated or wrong, so
  mixed-language recall on Latin-script references is 0 by construction. Decide whether ground truth
  should use Latin script (the current convention) or whether to score a transliteration-tolerant variant.
- **Whisper language forcing.** `generate_kwargs: {language: nepali}` stops Whisper from detecting Hindi,
  but can transliterate English. Test a variant with language unset for code-switching.
- **Numbers.** On the first real run, ASR returned "नव अर चौरासी करोड …" for a phone number, i.e. number
  words, not digits. Entity accuracy counts that as a miss, correctly for a voice agent. Closing this gap
  needs inverse text normalisation (ITN), which is not implemented yet.

## Telephone benchmark
`voice-lab benchmark telephone` runs every enabled ASR model through `telephone_suite` from
`configs/benchmark.yaml`:
- `clean_16k`
- `clean_8k`
- `telephone_8k`: 8 kHz, 300–3400 Hz band-pass, G.711 μ-law round trip
- `noisy_telephone_8k`: seeded white noise at 10 dB SNR before the line

Add a babble or background-speech profile with `add_noise: {noise_file: ...}` using audio you have rights
to. AMR/Opus codec simulation would need ffmpeg, which is not a dependency.

## Adding an ASR checkpoint
Any transformers ASR checkpoint works through `provider: hf-asr`. Add `generate_kwargs` for Whisper
models, or `long_form_kwargs: {chunk_length_s: 30}` for CTC models.
