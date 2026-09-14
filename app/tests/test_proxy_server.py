"""Real, end-to-end integration test of the Ramer proxy server -- against
the live Ollama on this machine, no mocks, per this project's own testing
discipline (see app/tests/test_scheduler.py for the pure-logic tests that
DO use a fake chat_impl; this file is deliberately the complementary real
half). Uses qwen2.5:3b-instruct, the smallest installed model, to keep the
suite fast.
"""
import json
import urllib.error
import urllib.request

from app.core.proxy_server import RamerProxyServer

MODEL = "qwen2.5:3b-instruct"


def _start_server() -> RamerProxyServer:
    server = RamerProxyServer(port=0, on_log=lambda msg: None)
    server.start_background()
    return server


def test_tags_passthrough_matches_real_ollama():
    server = _start_server()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/api/tags", timeout=10) as resp:
            data = json.loads(resp.read())
        names = [m["name"] for m in data["models"]]
        assert MODEL in names
    finally:
        server.stop()


def test_chat_streaming_relay_returns_real_content():
    server = _start_server()
    try:
        body = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with the single word: hello"}],
            "stream": True,
            "options": {"num_ctx": 2048, "num_predict": 10},
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/api/chat", data=body, headers={"Content-Type": "application/json"},
        )
        lines = []
        with urllib.request.urlopen(req, timeout=60) as resp:
            assert resp.status == 200
            for raw in resp:
                raw = raw.strip()
                if raw:
                    lines.append(json.loads(raw))
        assert len(lines) > 1, "expected multiple streamed NDJSON chunks, not one buffered blob"
        assert lines[-1].get("done") is True
        full_content = "".join(c.get("message", {}).get("content", "") for c in lines)
        assert len(full_content.strip()) > 0

        status = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{server.port}/ramer/status", timeout=10).read())
        assert status["requests_served"] == 1
        assert status["resident_model"] == MODEL
    finally:
        server.stop()


def test_chat_non_streaming_returns_single_json_object():
    server = _start_server()
    try:
        body = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with the single word: hello"}],
            "stream": False,
            "options": {"num_ctx": 2048, "num_predict": 10},
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/api/chat", data=body, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            assert resp.status == 200
            result = json.loads(resp.read())
        assert result["done"] is True
        assert len(result["message"]["content"].strip()) > 0
    finally:
        server.stop()


def test_openai_models_lists_real_ollama_models():
    server = _start_server()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/v1/models", timeout=10) as resp:
            data = json.loads(resp.read())
        assert data["object"] == "list"
        assert MODEL in [m["id"] for m in data["data"]]
    finally:
        server.stop()


def test_openai_chat_non_streaming_matches_openai_shape():
    server = _start_server()
    try:
        body = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with the single word: hello"}],
            "stream": False,
            "max_tokens": 10,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            assert resp.status == 200
            result = json.loads(resp.read())
        assert result["object"] == "chat.completion"
        choice = result["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert len(choice["message"]["content"].strip()) > 0
        assert choice["finish_reason"] == "stop"
        assert result["usage"]["completion_tokens"] > 0
    finally:
        server.stop()


def test_openai_chat_streaming_returns_sse_chunks_ending_in_done():
    server = _start_server()
    try:
        body = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with the single word: hello"}],
            "stream": True,
            "max_tokens": 10,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        raw_lines = []
        with urllib.request.urlopen(req, timeout=60) as resp:
            assert resp.status == 200
            assert "text/event-stream" in resp.headers.get("Content-Type", "")
            for line in resp:
                line = line.decode().strip()
                if line:
                    raw_lines.append(line)
        assert raw_lines[-1] == "data: [DONE]"
        chunks = [json.loads(l[len("data: "):]) for l in raw_lines[:-1]]
        assert all(c["object"] == "chat.completion.chunk" for c in chunks)
        full_content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert len(full_content.strip()) > 0
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    finally:
        server.stop()


def test_openai_chat_tool_call_arguments_are_json_string_not_object():
    """The one real shape difference from Ollama's native tool_calls --
    catching a regression here matters more than the other assertions."""
    server = _start_server()
    try:
        tools = [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the full contents of a file in the project directory.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            },
        }]
        body = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Use the read_file tool to read a file named notes.txt. "
                                                       "Only call the tool, do not respond with text."}],
            "tools": tools,
            "stream": False,
            "max_tokens": 60,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert tool_calls, "expected the model to call read_file"
        args = tool_calls[0]["function"]["arguments"]
        assert isinstance(args, str), "OpenAI wire format requires a JSON-encoded string, not a parsed object"
        parsed = json.loads(args)
        assert parsed["path"] == "notes.txt"
        assert result["choices"][0]["finish_reason"] == "tool_calls"
    finally:
        server.stop()


def test_openai_chat_replayed_tool_call_history_does_not_break_ollama():
    """Regression test for a real bug found via the `pi` CLI: a client
    replays its own prior assistant tool_calls (arguments as a JSON
    string, content: null) on the next turn -- exactly the shape this
    proxy's own /v1 responses produce. Sending that shape straight
    through to Ollama's native API breaks its request parsing; the proxy
    must convert arguments back to a parsed object and content back to a
    string before relaying."""
    server = _start_server()
    try:
        body = json.dumps({
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "Use the read_file tool to read notes.txt."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"path\": \"notes.txt\"}"},
                    }],
                },
                {"role": "tool", "content": "(file contents here)", "tool_call_id": "call_abc123"},
            ],
            "stream": False,
            "max_tokens": 30,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            assert resp.status == 200
            result = json.loads(resp.read())
        assert result["object"] == "chat.completion"
    finally:
        server.stop()


def test_unknown_path_returns_404():
    server = _start_server()
    try:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{server.port}/not/a/real/path", timeout=10)
            assert False, "expected an HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.stop()
