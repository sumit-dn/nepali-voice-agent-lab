# Licensing audit

Generated from `configs/*.yaml`. Every value was copied from the checkpoint's own card/licence on the `verified` date. **REVIEW REQUIRED** means we could not establish commercial rights - ask legal.

Terms are not interchangeable:

- **Open-source**: an OSI-approved licence (Apache-2.0, MIT, ...) on the weights *and* code.
- **Open-weight**: weights downloadable under a custom licence (Llama, Gemma ToU) with use restrictions.
- **Free**: no fee. Says nothing about commercial rights.
- **Commercially usable**: the exact licence of *this checkpoint* (and its training data, where it propagates) permits our commercial use.

## ASR

| ID | Checkpoint | Revision | Licence | Commercial use | Reference | Restrictions / notes | Verified |
|---|---|---|---|---|---|---|---|
| indicconformer-600m | ai4bharat/indic-conformer-600m-multilingual | e9b71b369c048e2c6b634d4c131061c34e441179 | MIT | yes | https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual | Card: 'released under the MIT license'. Training data not stated on card. Gated (auto-approve): accept terms + HF_TOKEN. | 2026-09-23 |
| indicconformer-600m-rnnt | ai4bharat/indic-conformer-600m-multilingual | e9b71b369c048e2c6b634d4c131061c34e441179 | MIT | yes | https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual | Same checkpoint as indicconformer-600m. | 2026-09-23 |
| indicconformer-ne-nemo | ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large | cd09ba7720f3b17d259f6bfd03e1463bc5ba517d | MIT | yes | https://huggingface.co/ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large | MIT per card. Gated (auto). Training data not stated. | 2026-09-23 |
| whisper-tiny | openai/whisper-tiny | 169d4a4341b33bc18d8881c4b69c2e104e1cc0af | MIT | yes | https://github.com/openai/whisper/blob/main/LICENSE | HF metadata says apache-2.0, GitHub says MIT for code+weights; both permissive. | 2026-09-23 |
| whisper-small | openai/whisper-small | 973afd24965f72e36ca33b3055d56a652f456b4d | MIT | yes | https://github.com/openai/whisper/blob/main/LICENSE | HF metadata apache-2.0 vs GitHub MIT; both permissive. | 2026-09-23 |
| whisper-large-v3-turbo | openai/whisper-large-v3-turbo | 41f01f3fe87f28c78e2fbf8b568835947dd65ed9 | MIT | yes | https://huggingface.co/openai/whisper-large-v3-turbo | Code and weights MIT (GitHub README). | 2026-09-23 |
| whisper-small-ne-dragneel | Dragneel/whisper-small-nepali | 72c458faca071ea24b84f566fb3439485df1e2f4 | Apache-2.0 | REVIEW REQUIRED | https://huggingface.co/Dragneel/whisper-small-nepali | Card licence Apache-2.0; trained on OpenSLR 54 (CC BY-SA 4.0) - whether ShareAlike reaches weights is unsettled. | 2026-09-23 |
| whisper-medium-ne-dragneel | Dragneel/whisper-medium-nepali-openslr | a435aa8b0879f7a7a5836be8926fb70315647313 | Apache-2.0 | REVIEW REQUIRED | https://huggingface.co/Dragneel/whisper-medium-nepali-openslr | Card Apache-2.0; data OpenSLR 54 (CC BY-SA 4.0). | 2026-09-23 |
| whisper-turbo-ne-sumanpaudel | sumanpaudel1997/nepali-asr-whisper-turbo | 00071400f501e001c1610b2921483353933aa14a | CC-BY-NC-4.0 | no | https://huggingface.co/sumanpaudel1997/nepali-asr-whisper-turbo | Research/non-commercial only. | 2026-09-23 |
| whisper-large-v3-ne-kiranpantha | kiranpantha/whisper-large-v3-nepali | 934c2bfe2955282afec7c7ad9271b3e9ce6c7587 | Apache-2.0 | REVIEW REQUIRED | https://huggingface.co/kiranpantha/whisper-large-v3-nepali | Card Apache-2.0; training set is private/removed (401) so provenance cannot be checked. | 2026-09-23 |
| indicwav2vec-ne-sumanpaudel | sumanpaudel1997/nepali-asr-indicwav2vec | aedaa1f860e9fec126748df81e8a9b5710038e60 | CC-BY-NC-4.0 | no | https://huggingface.co/sumanpaudel1997/nepali-asr-indicwav2vec | Research/non-commercial only. Base ai4bharat/indicwav2vec_v1_hindi. | 2026-09-23 |
| mms-1b-all-npi | facebook/mms-1b-all | 3d33597edbdaaba14a8e858e2c8caa76e3cec0cd | CC-BY-NC-4.0 | no | https://huggingface.co/facebook/mms-1b-all | Non-commercial. | 2026-09-23 |
| omniasr-ctc-1b | facebook/omniASR-CTC-1B | 8c22e3ffdaa4aab6431b128b84b991a7d9c2515c | Apache-2.0 | yes | https://github.com/facebookresearch/omnilingual-asr/blob/main/LICENSE | Code+models Apache-2.0; corpus CC-BY-4.0. | 2026-09-23 |
| qwen3-asr-1.7b | Qwen/Qwen3-ASR-1.7B | 7278e1e70fe206f11671096ffdd38061171dd6e5 | Apache-2.0 | yes | https://huggingface.co/Qwen/Qwen3-ASR-1.7B | Nepali NOT officially supported. | 2026-09-23 |
| qwen3-asr-ne-sidskarki | sidskarki/Qwen3-ASR-Nepali | 6ae9dd5306dde33ff2507820c626bac6fb867a7f | custom community-use-1.0 | REVIEW REQUIRED | https://huggingface.co/sidskarki/Qwen3-ASR-Nepali/blob/main/LICENSE.md | Only licensees with annual revenue <= USD 100k; larger orgs need a written licence. Not OSI. | 2026-09-23 |
| gemini-3.8-flash-asr | gemini-3.8-flash | – | Google Gemini API Terms | conditional | https://ai.google.dev/gemini-api/terms | Paid API. FREE TIER: Google may use submitted content to improve products and humans may review it - never send sensitive/personal data. | 2026-09-23 |
| gemini-3.5-transcribe | gemini-3.5-transcribe | – | Google Gemini API Terms | conditional | https://ai.google.dev/gemini-api/terms | As gemini-3.8-flash-asr. | 2026-09-23 |

## LLM

| ID | Checkpoint | Revision | Licence | Commercial use | Reference | Restrictions / notes | Verified |
|---|---|---|---|---|---|---|---|
| llama3.1-8b | llama3.1:latest | 46e0c10c039e | Llama 3.1 Community License | conditional | https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE | >700M MAU needs Meta licence; 'Built with Llama' attribution; derivative names start with 'Llama'; Acceptable Use Policy applies. | 2026-09-23 |
| qwen3-8b | qwen3:8b | 500a1f067a9f | Apache-2.0 | yes | https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE | Keep licence/NOTICE. | 2026-09-23 |
| qwen3-14b | qwen3:14b | bdbd181c33f2 | Apache-2.0 | yes | https://huggingface.co/Qwen/Qwen3-14B/blob/main/LICENSE |  | 2026-09-23 |
| qwen3-4b-instruct | qwen3:4b-instruct | – | Apache-2.0 | yes | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/LICENSE |  | 2026-09-23 |
| qwen3.5-9b | qwen3.5:9b | 6488c96fa5fa | Apache-2.0 | yes | https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE |  | 2026-09-23 |
| mistral-small-3.2-24b | mistral-small3.2:24b | 5a408ab55df5 | Apache-2.0 | yes | https://www.apache.org/licenses/LICENSE-2.0 |  | 2026-09-23 |
| gemma3-12b | gemma3:12b | – | Gemma Terms of Use | conditional | https://ai.google.dev/gemma/terms | Prohibited Use Policy incorporated; restrictions pass to redistributions/derivatives; Google may restrict usage. | 2026-09-23 |
| gemma4-12b | gemma4:12b | – | Apache-2.0 | yes | https://ai.google.dev/gemma/docs/gemma_4_license | Gemma 4 moved to Apache-2.0 (Gemma ToU explicitly excludes Gemma 4). | 2026-09-23 |
| gpt-oss-20b | gpt-oss:20b | – | Apache-2.0 | yes | https://huggingface.co/openai/gpt-oss-20b/blob/main/LICENSE | Plus USAGE_POLICY. | 2026-09-23 |
| aya-expanse-8b | aya-expanse:8b | – | CC-BY-NC-4.0 | no | https://creativecommons.org/licenses/by-nc/4.0/ | Plus Cohere Labs AUP. | 2026-09-23 |
| qwen3-8b-vllm | Qwen/Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 | Apache-2.0 | yes | https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE |  | 2026-09-23 |
| gemini-3.8-flash | gemini-3.8-flash | – | Google Gemini API Terms | conditional | https://ai.google.dev/gemini-api/terms | Free tier: content may be used to improve Google products and human-reviewed. | 2026-09-23 |
| gemini-3.5-flash-lite | gemini-3.5-flash-lite | – | Google Gemini API Terms | conditional | https://ai.google.dev/gemini-api/terms | As gemini-3.8-flash. | 2026-09-23 |

## TTS

| ID | Checkpoint | Revision | Licence | Commercial use | Reference | Restrictions / notes | Verified |
|---|---|---|---|---|---|---|---|
| piper-ne-google-x-low | rhasspy/piper-voices | c10ece1aade47bb51c153c893d14e5bf8e5b7117 | CC-BY-SA-4.0 (voice data); engine GPL-3.0-or-later | REVIEW REQUIRED | https://huggingface.co/rhasspy/piper-voices/resolve/main/ne/ne_NP/google/x_low/MODEL_CARD | Trained from scratch on OpenSLR-43 (Google, CC BY-SA 4.0): attribution + ShareAlike (reach into weights unsettled). piper-tts engine is GPL-3.0 - distribution implications need legal review. Cleanest Piper option. | 2026-09-23 |
| piper-ne-google-medium | rhasspy/piper-voices | c10ece1aade47bb51c153c893d14e5bf8e5b7117 | CC-BY-SA-4.0 (Nepali data) + lessac research-only lineage | REVIEW REQUIRED | https://huggingface.co/rhasspy/piper-voices/resolve/main/ne/ne_NP/google/medium/MODEL_CARD | Fine-tuned from en_US lessac; lessac/Blizzard-2013 licence is research-only and excludes commercial voice-synthesis development. | 2026-09-23 |
| piper-ne-chitwan-medium | rhasspy/piper-voices | c10ece1aade47bb51c153c893d14e5bf8e5b7117 | CC0 (data) + lessac research-only lineage | REVIEW REQUIRED | https://huggingface.co/rhasspy/piper-voices/resolve/main/ne/ne_NP/chitwan/medium/MODEL_CARD | Dataset CC0 (OHF-Voice voice-datasets) but weights fine-tuned from lessac (research-only data). | 2026-09-23 |
| indic-parler-tts | ai4bharat/indic-parler-tts | 7b527af5ee8ed1f9a28d80b19703ed9bb8ba10ca | Apache-2.0 | yes | https://huggingface.co/ai4bharat/indic-parler-tts | Card Apache-2.0; training data includes IndicTTS/LIMMITS/Rasa (CC BY 4.0) - attribute. Gated (auto): accept terms + HF_TOKEN. | 2026-09-23 |
| xtts-ne-oshara-e10 | Oshara/xtts-v2-nepali | 1ef72e4a13e201409a895ef45c36b38adbe324d8 | Coqui Public Model License 1.0.0 (inherited from coqui/XTTS-v2, per the fine-tune's own card) | no | https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt | CPML 'allows only non-commercial use of a machine learning model and its outputs'. Explicitly allowed: 'Use by commercial or for-profit entities for testing, evaluation, or non-commercial research and development' (so benchmarking here is fine). Not allowed: production/revenue use, or using outputs to train models for commercial use. Coqui shut down in 2024, so no commercial licence can be obtained. Fine-tune data ('4,140 single-speaker Nepali clips') source not stated - REVIEW REQUIRED. Runtime coqui-tts is MPL-2.0. | 2026-09-24 |
| xtts-ne-oshara-e20 | Oshara/xtts-v2-nepali | 1ef72e4a13e201409a895ef45c36b38adbe324d8 | Coqui Public Model License 1.0.0 (inherited from coqui/XTTS-v2, per the fine-tune's own card) | no | https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt | CPML 'allows only non-commercial use of a machine learning model and its outputs'. Explicitly allowed: 'Use by commercial or for-profit entities for testing, evaluation, or non-commercial research and development' (so benchmarking here is fine). Not allowed: production/revenue use, or using outputs to train models for commercial use. Coqui shut down in 2024, so no commercial licence can be obtained. Fine-tune data ('4,140 single-speaker Nepali clips') source not stated - REVIEW REQUIRED. Runtime coqui-tts is MPL-2.0. | 2026-09-24 |
| kala-nepali-v0.2 | ampixa/real-nepali-v0.2-kala | 90a66e818fbb4e19a8ba9b191da422a70e46a296 | CC-BY-SA-4.0 (card) vs MIT (PyPI kala-tts) | REVIEW REQUIRED | https://huggingface.co/ampixa/real-nepali-v0.2-kala | Licence conflict; ~half the training data is Gemini-generated synthetic; OpenSLR-143 is actually CC BY-NC-SA. | 2026-09-23 |
| matcha-tts-ne-sandipghimire | sandipghimire/matcha-tts-nepali | 2a7b77e62a3ff69340c7dc1aa93e886841d0cef6 | CC-BY-4.0 (weights) | REVIEW REQUIRED | https://huggingface.co/sandipghimire/matcha-tts-nepali | G2P frontend is CC-BY-SA-4.0; single author, 0 downloads, created 2026-07. | 2026-09-23 |
| omnivoice | k2-fsa/OmniVoice | c5fdb5ccb189668d56333f77ba2629f4cd7535f4 | CC-BY-NC (weights) | no | https://huggingface.co/k2-fsa/OmniVoice | Code Apache-2.0; weights non-commercial (Emilia data). | 2026-09-23 |
| gemini-3.1-flash-tts | gemini-3.1-flash-tts-preview | – | Google Gemini API Terms | conditional | https://ai.google.dev/gemini-api/terms | Preview model (no stable TTS exists). Free tier content may be used by Google. | 2026-09-23 |

## VAD

| ID | Checkpoint | Revision | Licence | Commercial use | Reference | Restrictions / notes | Verified |
|---|---|---|---|---|---|---|---|
| silero-vad | silero-vad (weights bundled in the pip package) | – | MIT | yes | https://github.com/snakers4/silero-vad/blob/master/LICENSE | README alt-text says CC BY-NC; badge and LICENSE file say MIT (alt-text is stale). | 2026-09-23 |

## Runtime dependencies (from installed package metadata)

| Package | Version | Licence |
|---|---|---|
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| scipy | 1.17.1 | BSD License |
| soundfile | 0.14.0 | BSD License |
| PyYAML | 6.0.3 | MIT License |
| psutil | 7.2.2 | BSD-3-Clause |
| torch | 2.14.0+cpu | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| torchaudio | 2.11.0+cpu | BSD License |
| transformers | 5.17.0 | Apache 2.0 License |
| huggingface_hub | 1.32.0 | Apache Software License |
| onnxruntime | 1.30.0 | MIT License |
| piper-tts | 1.8.0 | GPL-3.0-or-later |
| silero-vad | 6.2.1 | MIT License |

piper-tts is GPL-3.0-or-later: fine for internal research; shipping it inside a product needs legal review.
