# Architecture

## Layers
| Layer | Files | Responsibility |
|---|---|---|
| Registry | `configs/{asr,llm,tts,vad}.yaml`, `config.py` | Model entries. Unknown keys are rejected; each entry has a `fingerprint()` used for caching |
| Interfaces | `providers/base.py` | `ASRProvider.transcribe`, `LLMProvider.generate`, `TTSProvider.synthesize`, `VADProvider.detect_speech` and their result dataclasses |
| Adapters | `asr/`, `llm/`, `tts/`, `vad/` | One small class per runtime. They are imported lazily, so a missing heavy dependency only affects that adapter |
| Registry glue | `providers/registry.py` | Maps `provider:` names to classes, enforces the cloud guard, and selects enabled models |
| Audio | `audio/` | Decoding plus explicit preprocessing ops (mono, resample, bandpass, μ-law, noise, trim, normalise, denoise) |
| Evaluation | `evaluation/` | Text normalisation, WER/CER/alignment, entity accuracy, error tags, LLM validators, manifests |
| Benchmarking | `benchmarking/` | Generic runner, resource monitor, SQLite store, markdown reports |
| Pipeline | `pipeline.py` | Recorded-audio VAD → ASR → LLM → TTS with a timing waterfall |
| CLI | `cli.py` | The `voice-lab` entry point |

## Provider contract
- Adapters implement only the model call. The base class does load timing (`load()`, reported separately
  from inference), latency, RTF, and conversion to the model's input format.
- If a model needs 16 kHz mono and the audio is 8 kHz, the base class resamples it **and records that**
  in `metadata.model_input_conversion`. No audio conversion happens silently.
- TTS writes both the raw output and an evaluation copy.

## Benchmark flow
```
manifest ─► rows ─► for each Job(model, variant, config):
                     cached?(cache_key = model fingerprint + config hash, dataset sha, sample id) ─► reuse
                     else build provider ─► load (monitored) ─► evaluate(sample) ─► metrics
                   ─► results/<kind>/<run_id>/<model>__<variant>.json + summary.csv + samples.csv
                   ─► results/lab.db  (runs, models, samples)
```
A failing sample is recorded with its error and the run continues. A failing model (not downloaded,
server down, dependency missing) is recorded with a clear message, and the other models still run.

## Isolation guarantees (enforced in code)
- `HF_HUB_OFFLINE=1` is set for every command except `download`, so weights are never fetched implicitly.
- `external: true` models raise `CloudNotAllowed` unless `--allow-cloud` is passed, and print a WARNING when it is.
- Tools (`llm/tools.py`) are in-memory fakes with synthetic data.
- Nothing reads outside this repo except the shared HF cache (`~/.cache/huggingface`, which holds public
  weights) and the local Ollama daemon. Set `HF_HOME=./.hf-cache` to isolate the cache as well.

## Adding a model
1. Add an entry to the right `configs/*.yaml` (see existing ones). Fill in the licence from **its own** card.
2. If the runtime is new, add an adapter (usually 20–40 lines) and register it in `ADAPTERS`.
3. Run `voice-lab download <id>`, then `voice-lab benchmark <kind> --model <id>`.
