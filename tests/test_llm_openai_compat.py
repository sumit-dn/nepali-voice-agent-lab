"""OpenAI-compatible client against a local fake SSE server (no Ollama needed)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from voice_lab.config import ModelSpec
from voice_lab.llm.openai_compat import OpenAICompatLLM
from voice_lab.llm.tools import call_tool, run_with_tools
from voice_lab.providers.base import ProviderError


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output quiet
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json({"data": [{"id": "fake-model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert req["stream"] is True and req["model"] == "fake-model"
        answered_tool = any(m["role"] == "tool" for m in req["messages"])
        if req.get("tools") and not answered_tool:
            chunks = [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c1",
                                        "function": {"name": "get_balance", "arguments": '{"customer'},
                                    }
                                ]
                            }
                        }
                    ]
                },
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '_id": "C-1001"}'}}]}}]},
            ]
        else:
            chunks = [
                {"choices": [{"delta": {"content": "<think>hmm</think>तपाईंको "}}]},
                {"choices": [{"delta": {"content": "balance रु. २५,००० छ।"}}]},
            ]
        chunks.append({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 7}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for c in chunks:
            self.wfile.write(f"data: {json.dumps(c, ensure_ascii=False)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def spec(url: str, model: str = "fake-model") -> ModelSpec:
    return ModelSpec(
        id="t", provider="openai-compat", checkpoint=model, params={"default_base_url": url, "api_prefix": "/v1"}
    )


def test_stream_text_usage_and_think_stripped(server):
    llm = OpenAICompatLLM(spec(server))
    r = llm.generate([{"role": "user", "content": "hi"}])
    assert r.text == "तपाईंको balance रु. २५,००० छ।"
    assert r.metadata["raw_text"].startswith("<think>")
    assert r.output_tokens == 7 and r.input_tokens == 12
    assert r.time_to_first_token is not None and r.time_to_first_token <= r.latency_seconds


def test_tool_loop_uses_fake_tools(server):
    final, trace, ttft, total = run_with_tools(OpenAICompatLLM(spec(server)), [{"role": "user", "content": "balance?"}])
    assert trace[0]["name"] == "get_balance" and trace[0]["arguments"] == {"customer_id": "C-1001"}
    assert trace[0]["result"]["balance"] == 25000
    assert "२५,०००" in final.text and ttft is not None and total > 0


def test_health_and_missing_model(server):
    assert OpenAICompatLLM(spec(server)).health()["ok"] is True
    with pytest.raises(ProviderError, match="not served"):
        OpenAICompatLLM(spec(server, "absent")).load()


def test_unreachable_server_is_clear():
    with pytest.raises(ProviderError, match="Cannot reach"):
        OpenAICompatLLM(spec("http://127.0.0.1:9")).health()


def test_fake_tools_reject_bad_args():
    assert call_tool("get_balance", {"wrong": 1})["error"].startswith("bad_arguments")
    assert call_tool("nope", {})["error"] == "unknown_tool:nope"
