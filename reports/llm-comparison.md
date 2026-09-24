# LLM comparison

> Evidence only. This report does not rank models or pick a winner. Weigh quality, latency, hardware, concurrency and licensing for the actual use case. Numbers come only from runs listed below.

- Run: `/home/digital-nepal/Documents/dn-ai/nepali-voice-agent-lab/results/llm/20260923T102719627Z-llm`
- Dataset: `data/manifests/llm.jsonl` (sha d763397ff7bf)
- Hardware: Intel(R) Core(TM) i5-7400 CPU @ 3.00GHz (4 threads), 16.1 GB RAM, GPU: none (CPU-only)

## Deterministic checks

| Model | Checkpoint | N | Failed | Pass rate | contains | contains_any | max_devanagari | max_words | min_devanagari | no_tool | not_contains | tool_args | tool_name | Hindi-leak (heuristic) | TTFT p50 (s) | Latency p50 (s) | Tok/s | License | Commercial | Error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama3.1-8b | llama3.1:latest | 22 | 0 | 0.842 | 0.917 | 0.333 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.833 | 0.857 | 0.045 | 4.419 | 24.403 | 2.600 | Llama 3.1 Community License | conditional | – |

## Needs human evaluation

Automatic checks cannot judge fluency/naturalness. Rate these replies 1-5.

- **llama3.1-8b / gen-greeting** (nepali_quality, naturalness): मेरो नमस्कार, मैले ठीक छु। कुन कामले मेरो सहायता लिन चाहान?
- **llama3.1-8b / gen-thanks** (nepali_quality): नमस्कार! तपाईंको नाम के के हो?
- **llama3.1-8b / mix-book-tomorrow** (mixed_language_handling): तपाईंको appointment tomorrow को १० बजे लागि सफलतापूर्वक book गरिएको छ। Appointment ID A-777 हो।
- **llama3.1-8b / inst-brevity** (conciseness_for_voice): हाम्रो Sagarmatha Services मा चार प्रकारका internet plans छन्: होम ५० Mbps रु. १,२०० प्रति महिना, होम १०० Mbps रु. १,८०० प्रति महिना, बिजネसमा २०० Mbps रु. ४,५०० प्रति महिना।
