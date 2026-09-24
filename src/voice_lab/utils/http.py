"""Tiny stdlib HTTP helper so HTTP providers need no extra dependency."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from http.client import HTTPResponse
from typing import Any

from voice_lab.providers.base import ProviderError


def open_url(
    url: str,
    *,
    body: Any = None,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> HTTPResponse:
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read()[:400].decode("utf-8", "replace")
        raise ProviderError(f"HTTP {e.code} from {url.split('?')[0]}: {detail}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        reason = getattr(e, "reason", e)
        raise ProviderError(f"Cannot reach {url.split('?')[0]} ({reason}). Is the server running?") from e


def get_json(url: str, **kwargs: Any) -> Any:
    with open_url(url, **kwargs) as resp:
        return json.load(resp)
