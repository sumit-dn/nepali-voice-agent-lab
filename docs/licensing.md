# Licensing

`reports/licensing.md` is generated from the registries by `voice-lab reports`, so it always matches the
configs. This page summarises the questions a lawyer needs to answer.

## Rules followed
- The licence is read from **the exact checkpoint's own card**, never inferred from a base model or engine.
- Entries marked `REVIEW REQUIRED` are unresolved; do not treat them as usable commercially.
- The verification date and source URLs are stored in every entry.

## Open questions (REVIEW REQUIRED)
1. **CC BY-SA training data → weights.**
   - The affected models: every OpenSLR-54-trained Nepali Whisper, and Piper `google-*` (trained on OpenSLR-43).
   - The question: does ShareAlike attach to model weights?
2. **Piper `*-medium` voices:** these were fine-tuned from `en_US-lessac`, whose Blizzard-2013 data licence
   excludes commercial voice-synthesis development.
3. **piper-tts engine is GPL-3.0-or-later** (the `rhasspy/piper` MIT code is archived). What are the
   distribution obligations if it ships inside a product?
4. **sidskarki/Qwen3-ASR-Nepali:** under its custom licence, only organisations with annual revenue ≤ USD
   100k may use it.
5. **kala-nepali:** the HF card says CC-BY-SA while PyPI says MIT; part of its data is CC BY-NC-SA and part
   is Gemini-generated.
6. **Llama 3.1 / Gemma 3:** these are open-weight, not open-source. There are attribution and AUP
   obligations, and Gemma's terms allow Google to restrict use remotely.
7. **Gemini free tier:** submitted content may be used by Google and reviewed by humans.

## Clean so far, pending the org's own review
- IndicConformer-600M (MIT; training data not stated)
- Whisper (MIT)
- Qwen3 8B/14B, Mistral Small 3.2, Gemma 4 (Apache-2.0)
- Indic Parler-TTS (Apache-2.0; attribute its CC-BY training sets)
- Silero VAD (MIT)
- omniASR (Apache-2.0)
