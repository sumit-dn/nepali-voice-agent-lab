# LLM lab

## Serving
Every local LLM goes through one OpenAI-compatible client (`llm/openai_compat.py`, stdlib only):

| server | `params` | health |
|---|---|---|
| Ollama | `server: ollama`, `base_url_env: OLLAMA_BASE_URL`, `api_prefix: /v1` | `/v1/models`, plus `/api/version` and `/api/tags` for quantisation and size |
| vLLM | `server: vllm`, `base_url_env: VLLM_BASE_URL` | `/v1/models` (vLLM also exposes `/health`) |
| Gemini (cloud) | `api_key_env: GEMINI_API_KEY`, `external: true` | `/v1beta/openai/models` |

The same model on a different server is simply another registry entry (`qwen3-8b` vs `qwen3-8b-vllm`).
Requests always stream:
- **TTFT** is the time to the first *speakable* content or tool call. Reasoning tokens are timed
  separately (`time_to_first_reasoning_token`).
- Output tokens come from the server's `usage` field, requested with `stream_options.include_usage`.
- `<think>…</think>` is stripped from the text; the raw text is kept in the metadata.

## Candidates (see configs/llm.yaml)
| Model | Nepali official? | Licence | Q4 size |
|---|---|---|---|
| Qwen3-8B / 14B | **yes** (Qwen3 language list) | Apache-2.0 | 5.2 / 9.3 GB |
| Mistral Small 3.2 24B | **yes** (3.1 card) | Apache-2.0 | 15 GB |
| Gemma 4 12B | not listed (140+ claimed) | Apache-2.0 | 7.6 GB |
| Gemma 3 12B | not listed | Gemma ToU (open-weight, conditional) | 8.1 GB |
| Llama 3.1 8B | **no** | Llama 3.1 Community (conditional) | 4.9 GB, installed locally |
| Qwen3-4B-Instruct-2507 | family claim | Apache-2.0 | 2.5 GB, CPU-class |

**Thinking mode:** Qwen3 thinks by default. Entries send `extra_body: {reasoning_effort: "none"}` for
Ollama and `chat_template_kwargs: {enable_thinking: false}` for vLLM. **This is untested here because Qwen
is not pulled yet.** Check that `reasoning_chars` in the results metadata is 0.

## Test set (data/manifests/llm.jsonl)
There are 22 cases:
- general Nepali, customer support, mixed language
- numbers (including the 11-digit repeat test), currency arithmetic
- Bikram Sambat dates, names
- long context against a fictional knowledge base
- a hallucination probe (a branch that does not exist), English replies
- instruction following, tool calling (all six fake tools), refusal

Validators are deterministic (`evaluation/llm_checks.py`): tool name and arguments, substring and digit
checks, Devanagari share, and word limits. Cases tagged `human_eval` get listed in the report for 1–5
human rating.

## First observation (Llama 3.1 8B Q4, CPU)
TTFT was 33–40 s. In the tool-calling smoke test the model invented a phone number, never called
`get_balance`, and hallucinated a balance (20,000 against the true 25,000). These are exactly the failure
modes the validators are there to catch. See `reports/llm-comparison.md`.
