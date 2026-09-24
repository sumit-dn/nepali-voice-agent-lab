"""Shared Gemini REST call (generateContent). Only reachable through models marked `external: true`."""

from __future__ import annotations

import json
import os
from typing import Any

from voice_lab.providers.base import ProviderError
from voice_lab.utils.http import open_url

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ProviderError("GEMINI_API_KEY is not set. Add a personal/test key to .env (never a production key).")
    return key


def generate_content(model: str, body: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    with open_url(
        f"{BASE_URL}/models/{model}:generateContent",
        body=body,
        headers={"x-goog-api-key": api_key()},
        timeout=timeout,
    ) as resp:
        return json.load(resp)


def parts(response: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return response["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError) as e:
        raise ProviderError(f"Unexpected Gemini response: {json.dumps(response)[:300]}") from e
