# Deployment

**Status: not deployed, on purpose.** Local experimentation comes first (Step 36).

## Order of work
1. Milestones 1–2 (done in this first pass):
   - component labs and benchmarks
   - the recorded-audio VAD → ASR → LLM → TTS pipeline
2. Milestone 3:
   - streaming ASR, LLM and TTS, and time-to-first-audio
   - barge-in simulation on recorded audio
   - concurrency and load tests
   - quantisation comparisons
3. Then containers, kept to as few as possible:
   - `docker/llm`: stock `vllm/vllm-openai` or Ollama images; nothing custom is needed, because the lab
     already talks to them over HTTP.
   - `docker/asr`, `docker/tts`: only once a model is chosen, as small HTTP services exposing the same
     provider interface.
4. Only after that, think about telephony.

## Future telephony integration (design notes, not implemented)
- A separate gateway service would receive 8 kHz μ-law media (RTP or WebSocket) from a **test** PBX and run
  streaming VAD → ASR → LLM → TTS, returning audio.
- It would reuse the provider interfaces and the telephone preprocessing profiles, so lab results carry over.
- Prerequisites:
  - streaming benchmarks
  - barge-in handling
  - measured concurrency limits
  - a fallback policy (local-first, optional cloud fallback, off by default)
  - a signed-off licensing review
- Never connect it to the production PBX or media endpoints from this repo.
