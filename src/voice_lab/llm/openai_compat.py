"""OpenAI-compatible chat client: Ollama, vLLM, and Gemini's /openai endpoint. Stdlib only.

Streams every request so time-to-first-token is measured the same way on every server.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from voice_lab.providers.base import LLMProvider, LLMResult, ProviderError
from voice_lab.utils.http import get_json, open_url

_THINK = re.compile(r"<think>.*?</think>", re.S)


def _parse_args(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw) if raw.strip() else {}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    except json.JSONDecodeError:
        return {"_unparsed": raw}


class OpenAICompatLLM(LLMProvider):
    def __init__(self, spec: Any, device: str | None = None):
        super().__init__(spec, device)
        p = spec.params
        root = os.environ.get(p.get("base_url_env", ""), "") or p.get("default_base_url", "")
        if not root:
            raise ProviderError(f"{spec.id}: set params.base_url_env / default_base_url in configs/llm.yaml")
        self.root = root.rstrip("/")
        self.base_url = self.root + p.get("api_prefix", "")
        self.api_key = os.environ.get(p["api_key_env"]) if p.get("api_key_env") else None
        if p.get("api_key_env") and not self.api_key:
            raise ProviderError(f"{spec.id} needs {p['api_key_env']} in .env (use a personal/test key).")
        self.timeout = float(p.get("timeout", 300))

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def health(self) -> dict[str, Any]:
        start = time.perf_counter()
        listed = get_json(f"{self.base_url}/models", headers=self._headers(), timeout=10)
        ids = [m["id"] for m in listed.get("data", [])]
        want = self.spec.checkpoint
        info: dict[str, Any] = {
            "server": self.base_url,
            "model": want,
            "model_available": want in ids or f"models/{want}" in ids,
            "response_seconds": round(time.perf_counter() - start, 3),
        }
        if self.spec.params.get("server") == "ollama":  # extra detail only Ollama exposes
            info["server_version"] = get_json(f"{self.root}/api/version", timeout=10).get("version")
            for m in get_json(f"{self.root}/api/tags", timeout=10).get("models", []):
                if m.get("name") == want or m.get("model") == want:
                    info.update(
                        quantization=m["details"].get("quantization_level"),
                        parameter_size=m["details"].get("parameter_size"),
                        size_gb=round(m.get("size", 0) / 1e9, 2),
                    )
        info["ok"] = info["model_available"]
        if not info["ok"]:
            info["available_models"] = ids[:30]
        return info

    def _load(self) -> None:
        info = self.health()
        if not info["ok"]:
            hint = (
                f" Pull it first: ollama pull {self.spec.checkpoint}"
                if self.spec.params.get("server") == "ollama"
                else ""
            )
            raise ProviderError(
                f"{self.spec.checkpoint} is not served at {self.base_url}.{hint} (size: {self.spec.size})"
            )

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        self.load()
        p = self.spec.params
        body: dict[str, Any] = {
            "model": self.spec.checkpoint,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": p.get("temperature", 0.0) if temperature is None else temperature,
            "max_tokens": max_tokens or p.get("max_tokens", 512),
            **p.get("extra_body", {}),
        }
        if tools:
            body["tools"] = tools
        start = time.perf_counter()
        ttft = first_reasoning = None
        text: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        with open_url(
            f"{self.base_url}/chat/completions", body=body, headers=self._headers(), timeout=self.timeout
        ) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    now = time.perf_counter() - start
                    if think := delta.get("reasoning") or delta.get("reasoning_content"):
                        first_reasoning = first_reasoning or now
                        reasoning.append(think)
                    if delta.get("content") or delta.get("tool_calls"):
                        ttft = ttft or now  # first *speakable* output; reasoning tokens don't count
                    text.append(delta.get("content") or "")
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", len(calls)), {"id": None, "name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        slot["id"] = slot["id"] or tc.get("id")
                        slot["name"] = slot["name"] or fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
        raw_text = "".join(text)
        return LLMResult(
            text=_THINK.sub("", raw_text).strip(),
            model=self.spec.id,
            provider=self.spec.provider,
            latency_seconds=time.perf_counter() - start,
            time_to_first_token=ttft,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            tool_calls=[
                {"id": c["id"] or f"call_{i}", "name": c["name"], "arguments": _parse_args(c["arguments"])}
                for i, c in sorted(calls.items())
            ],
            metadata={
                "server": self.base_url,
                "checkpoint": self.spec.checkpoint,
                "raw_text": raw_text if raw_text != _THINK.sub("", raw_text) else None,
                "reasoning_chars": len("".join(reasoning)),
                "time_to_first_reasoning_token": first_reasoning,
            },
        )
