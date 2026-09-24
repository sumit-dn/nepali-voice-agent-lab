# Data

Everything under `data/raw/`, `data/processed/`, `data/synthetic/` and `data/noise/` is git-ignored.
Only manifests with **synthetic, company-neutral text** are committed. Never put production calls,
customer recordings or company documents here.

## Manifests (`data/manifests/*.jsonl`, one JSON object per line)

### ASR — `asr.jsonl` (you create this from real recordings)

| field | required | notes |
|---|---|---|
| `id` | yes | unique |
| `audio` | yes | path relative to the repo root (WAV/FLAC/OGG/MP3) |
| `text` | yes | verified ground-truth transcript. Nepali in Devanagari; **English words in Latin script** |
| `language` | | `ne`, `en`, `ne-en` |
| `category` | | see the category list below |
| `speaker_id` | | pseudonymous id, never a real name |
| `sample_rate` | | as recorded |
| `entities` | | critical entities to score: `phone_numbers`, `otps`, `account_numbers`, `numbers`, `currency` (compared as digit strings), `dates`, `times`, `names`, `addresses` (compared as normalised phrases) |
| `speaker_gender`, `accent`, `speaking_rate`, `noise`, `channel` | | recording conditions, used to slice results |
| `consent`, `deidentified` | | set both to `true` before a recording enters any benchmark |
| `source` | | `recorded`, `public:<dataset>`, or `synthetic-tts:<model>` |

See `asr.example.jsonl`. Validate with `voice-lab dataset validate data/manifests/asr.jsonl --kind asr`.

**Ground-truth rules.** Transcribe what was said, not what was meant. Write digits the way the speaker said them:
use digits if the speaker read digits, and words if they said words. Keep English words in Latin script. Never
auto-generate ground truth with an ASR model.

### Categories

Content categories come from `asr_prompts.jsonl`, the reading script for recording sessions: basic, long,
Devanagari, conjuncts, numbers, dates, currency, names, addresses, proper nouns, English, mixed, and
conversational. Recording-condition categories are captured with the condition fields above rather than
with `category`: male/female, accents, fast/slow, background noise, telephone channel, long-form, and
interruptions.

### LLM — `llm.jsonl`
`{id, category, prompt | messages, tools?, context_file?, expect{...}, human_eval?[...]}`. The validators
are documented in `src/voice_lab/evaluation/llm_checks.py`. The tools are fake and local (`src/voice_lab/llm/tools.py`),
and `context_file` points at the fictional knowledge base in `data/kb/`.

### TTS — `tts.jsonl`
`{id, category, text}`. These are input texts only; there is no ground truth to fabricate.

## Synthetic audio
`voice-lab dataset synth --tts piper-ne-google-x-low` reads `asr_prompts.jsonl`, generates audio, and writes
`asr_synthetic_<model>.jsonl` with `source: synthetic-tts:<model>`. Use it for **pipeline smoke tests only**.
TTS audio is not real speech, so ASR scores on it must never drive model selection.

## Public datasets worth importing
Check each licence before use:
- FLEURS `ne_np` (CC-BY-4.0)
- OpenSLR 54 (CC BY-SA 4.0)
- OpenSLR 43 (CC BY-SA 4.0)
- Common Voice: the licence is now REVIEW REQUIRED; it is distributed through the Mozilla Data Collective.

Most Nepali fine-tunes were trained on OpenSLR 54, so scoring those models on OpenSLR 54 overstates their quality.
