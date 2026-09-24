# TTS lab

## Candidates (configs/tts.yaml, verified 2026-09-23)
| id | runtime | rate | licence status |
|---|---|---|---|
| piper-ne-google-x-low | piper (ONNX, CPU) | 16 kHz, 18 female speakers | CC-BY-SA-4.0 data, trained from scratch. **REVIEW REQUIRED** (ShareAlike; GPL-3.0 engine) |
| piper-ne-google-medium | piper | 22.05 kHz | **REVIEW REQUIRED**: fine-tuned from lessac, whose data licence is research-only |
| piper-ne-chitwan-medium | piper | 22.05 kHz, 1 speaker | **REVIEW REQUIRED**: CC0 data, lessac lineage |
| indic-parler-tts | parler (GPU) | 44.1 kHz, Nepali voice "Amrita" | Apache-2.0, gated |
| gemini-3.1-flash-tts | cloud | 24 kHz | preview model, off by default |

Checked and absent:
- `facebook/mms-tts-npi` does not exist, and Nepali is not in the MMS-TTS list.
- Chatterbox, IndicF5, Kokoro and XTTS-v2 have no official Nepali.

Community Matcha/Kala/OmniVoice checkpoints are tracked but have licence problems or no adoption.

## Findings from the first run (CPU)
- All three Piper voices run at RTF ≈ 0.08–0.11 and load in under 1.5 s.
- **Aspiration dropped.** Piper logs `Missing phoneme from id map: ʰ`: the espeak-ng Nepali front-end emits
  aspiration marks the voice cannot render, so aspirated consonants (ख घ छ झ ठ ढ थ ध फ भ) are likely
  degraded. Confirm by listening.
- **No text normalisation.** Piper read `९८४१२३४५६७` as one cardinal number ("nine arab eighty-four crore …").
  A voice agent needs a Nepali text normaliser before TTS: digit-by-digit phone numbers and OTPs, currency,
  BS dates, times. Not implemented yet.

## Indic Parler-TTS (separate environment)
`parler-tts` pins `transformers==4.46.1`, which conflicts with the main env (transformers 5.x). Use a second venv:
```bash
uv venv .venv-parler --python 3.11 && source .venv-parler/bin/activate
uv pip install -e . "git+https://github.com/huggingface/parler-tts.git" torch soundfile
HF_TOKEN=... voice-lab download indic-parler-tts     # 3.76 GB, gated
voice-lab tts synthesize --model indic-parler-tts --text "नमस्ते" --device cuda
```
On CPU it will run but slowly; plan for a GPU.

## Oshara XTTS-v2 Nepali (separate environment)
The checkpoint is `Oshara/xtts-v2-nepali`, a full fine-tune of Coqui XTTS-v2 with zero-shot voice cloning at 24 kHz. It is registered as
`xtts-ne-oshara-e10` (the card recommends it) and `xtts-ne-oshara-e20`. Both are off by default.

- **Licence: non-commercial (CPML).** Testing and evaluation by a company is explicitly allowed. Production use, and training
  commercial models on its output, are not. Coqui no longer exists to sell a commercial licence.
- **Environment:** `make install-xtts` builds `.venv-xtts`. It is a separate environment for two reasons:
  - coqui-tts 0.27.5 imports an API that transformers 5 removed;
  - with torch ≥ 2.9 it would need torchcodec and FFmpeg, so the environment pins torch 2.8.

  The lab drives it through `tts/xtts_worker.py` over JSON lines, so the CLI, UI and benchmarks use it like any other model.
- **Language route:** the card says `language="ne"`, but the vocabulary has no `[ne]` token. On that route the model reads
  "ने" aloud before every sentence (ASR-judge CER 0.094). The adapter uses the `[hi]` token instead (CER 0.000 on the same
  sentences). As a side effect, digits are not read as Nepali: `१०` was heard as "एइस". Normalise numbers before TTS.
- **Voices:** 58 built-in XTTS speakers (default `Claribel Dervla`), or clone from a reference WAV (in the UI, the TTS tab's reference box).
  **Clone only voices whose owners consented.**
- **Speed on the i5-7400:** RTF about 3.8, about 17 s per sentence, 28 s to load (173 s when running next to other large models).
  A GPU is needed for anything interactive.

## Output format
Each synthesis writes `<id>.orig.wav`, the model's raw samples at their native dtype and rate, and
`<id>.wav`, a mono 16-bit PCM copy for evaluation. The input text, model, voice, latency, duration and
sample rate are recorded in the run JSON.
