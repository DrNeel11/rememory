"""Ramer Proxy Server: the actual shape of "invisible infrastructure"
discussed in PRODUCT_HYPOTHESES.md -- an HTTP server that speaks Ollama's
own wire protocol (/api/chat, /api/generate, /api/tags, /api/show) AND the
OpenAI-compatible chat API (/v1/chat/completions, /v1/models), so a
developer points their EXISTING agent tool (Cline, Continue, aider,
mini_agent.py, the `pi` CLI, or anything else that already knows how to
talk to Ollama or an OpenAI-style endpoint) at Ramer's port instead of
Ollama's, with zero change to that tool. From the tool's point of view,
nothing changed. Underneath:

  developer's agent tool
        |
   Ramer proxy (this file)
        |
    RamerScheduler   <-- priority + sticky residency (app/core/scheduler.py)
        |
    PlacementEngine  <-- per-model num_gpu/num_ctx tuning (app/core/placement.py)
        |
    real Ollama

Only /api/chat requests go through the scheduler -- that's the real target
(tool-calling coding agents all use it) and the thing the scheduler's
priority/stickiness was built and validated for. /api/generate is relayed
directly (still placement-tuned, not scheduler-queued): a v0 scope choice,
not an oversight -- add scheduling to it if a real client turns out to
need it. /api/tags and /api/show are pure metadata passthroughs.

num_gpu is resolved per-request from PlacementEngine's cache (keyed by
model+hardware+num_ctx, same cache ramer_local.py/`ramer benchmark`
writes) unless the caller's own `options.num_gpu` already specifies one.
Deliberately does NOT trigger a probe inline on a cache miss -- a probe
takes 1-3 minutes of real wall-clock sweeping, which would make an
interactive agent request hang unacceptably; falls back to Ollama's own
"auto" and logs that a one-time `ramer benchmark <model>` would help.

/v1/chat/completions translates OpenAI's request/response shapes to and
from the same scheduler+placement path /api/chat uses -- discovered
necessary in practice, not designed in speculatively: several real
agent tools (the `pi` CLI among them) configure their "ollama" provider
against Ollama's OpenAI-compatible surface (http://host:11434/v1)
rather than its native API, so a proxy that only speaks the native API
silently 404s for them. Tool-call arguments are the one real format
difference -- Ollama hands them back as a parsed object, OpenAI's wire
format wants them as a JSON-encoded string -- everything else
(messages, tool schemas) is already shape-compatible between the two.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from app.core.placement import PlacementEngine
from app.core.scheduler import RamerScheduler

DEFAULT_UPSTREAM = "http://127.0.0.1:11434"  # real Ollama; NOT "localhost" -- see ramer_local.py's note on the ~2s DNS stall
DEFAULT_PROXY_PORT = 11435

PASSTHROUGH_GET_PATHS = {"/api/tags", "/api/version", "/api/ps"}
PASSTHROUGH_POST_PATHS = {"/api/show"}


def _relay(endpoint: str, model: str, payload_extra: dict, num_gpu, num_ctx: int,
           num_predict: Optional[int], upstream: str, on_chunk: Optional[Callable[[dict], None]] = None) -> dict:
    """POSTs a streaming request to real Ollama and relays each raw NDJSON
    chunk to on_chunk as it arrives (so a client waiting on the proxy sees
    tokens as they're generated, same as talking to Ollama directly), while
    accumulating the final message/response for whoever is blocked on the
    return value (RamerScheduler.submit, or the proxy handler for
    /api/generate). endpoint is "/api/chat" or "/api/generate" -- their
    streamed chunk shapes differ (message.content vs response), handled
    here rather than duplicating this function per endpoint."""
    options = {"num_ctx": num_ctx}
    if num_gpu != "auto":
        options["num_gpu"] = num_gpu
    if num_predict is not None:
        options["num_predict"] = num_predict
    body = dict(payload_extra, model=model, stream=True, options=options)
    req = urllib.request.Request(
        f"{upstream}{endpoint}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    content_parts: list[str] = []
    tool_calls: list = []
    final: dict = {}
    try:
        resp_ctx = urllib.request.urlopen(req, timeout=300)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise urllib.error.HTTPError(e.url, e.code, f"{e.reason}: {detail[:500]}", e.headers, None) from None
    with resp_ctx as resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if on_chunk is not None:
                on_chunk(chunk)
            if endpoint == "/api/chat":
                msg = chunk.get("message", {})
                if msg.get("content"):
                    content_parts.append(msg["content"])
                if msg.get("tool_calls"):
                    tool_calls.extend(msg["tool_calls"])
            else:
                if chunk.get("response"):
                    content_parts.append(chunk["response"])
            if chunk.get("done"):
                final = chunk
                break
    if endpoint == "/api/chat":
        message = {"role": "assistant", "content": "".join(content_parts)}
        if tool_calls:
            message["tool_calls"] = tool_calls
        final["message"] = message
    else:
        final["response"] = "".join(content_parts)
    return final


def _normalize_openai_messages(messages: list) -> list:
    """Two real shape mismatches between OpenAI's request format and
    Ollama's native one -- both found by capturing real payloads from the
    `pi` CLI, not guessed in advance:

    1. Message content may be a list of parts (e.g. [{"type": "text",
       "text": "..."}]) instead of a plain string -- valid for OpenAI,
       rejected by Ollama, which requires a string. An assistant message
       that made a tool call often carries content: null for the same
       reason -- also needs to become "".
    2. A replayed assistant message's tool_calls[].function.arguments is
       a JSON-encoded STRING in OpenAI's format (the exact shape this
       proxy's own /v1 responses produce, so a client replaying its own
       history sends this back to us) -- but Ollama's native schema wants
       a parsed object there. Sending the string back verbatim breaks
       Ollama's own request parsing with a cryptic "value looks like
       object, but can't find closing '}' symbol" error.
    """
    normalized = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            m["content"] = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
        elif content is None:
            m["content"] = ""
        if m.get("tool_calls"):
            fixed_calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function", {}))
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        pass  # not the shape expected -- leave as-is rather than fail the whole request over it
                tc["function"] = fn
                fixed_calls.append(tc)
            m["tool_calls"] = fixed_calls
        normalized.append(m)
    return normalized


def _openai_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _ollama_tool_calls_to_openai(tool_calls: Optional[list]) -> list:
    """Ollama hands back parsed-object arguments; OpenAI's wire format
    wants a JSON-encoded string in that field -- the one real shape
    difference between the two tool-call formats."""
    out = []
    for i, tc in enumerate(tool_calls or []):
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args)
        out.append({
            "index": i,
            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": fn.get("name", ""), "arguments": args},
        })
    return out


def _openai_stream_chunk(chat_id: str, created: int, model: str, delta: dict, finish_reason: Optional[str]) -> dict:
    return {
        "id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "RamerProxy/0.1"

    def log_message(self, fmt, *args) -> None:  # noqa: A002 -- stdlib override; route through ramer.log instead of stderr
        self.server.ramer.log("%s - %s" % (self.address_string(), fmt % args))

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw) if raw.strip() else {}

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, code: int, message: str) -> None:
        self._send_json(code, {"error": message})

    def _proxy_passthrough(self, method: str) -> None:
        upstream = self.server.ramer.upstream
        data = None
        headers = {}
        if method == "POST":
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length) if length else b"{}"
            headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(f"{upstream}{self.path}", data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.URLError as e:
            self._send_error_json(502, f"upstream Ollama unreachable: {e}")

    def do_GET(self) -> None:
        if self.path in PASSTHROUGH_GET_PATHS:
            self._proxy_passthrough("GET")
        elif self.path == "/v1/models":
            self._handle_openai_models()
        elif self.path == "/ramer/status":
            self._send_json(200, self.server.ramer.status())
        else:
            self._send_error_json(404, f"unhandled path {self.path}")

    def _handle_openai_models(self) -> None:
        ramer = self.server.ramer
        try:
            with urllib.request.urlopen(f"{ramer.upstream}/api/tags", timeout=10) as resp:
                tags = json.loads(resp.read())
        except urllib.error.URLError as e:
            self._send_error_json(502, f"upstream Ollama unreachable: {e}")
            return
        data = [{"id": m["name"], "object": "model", "created": 0, "owned_by": "ollama"} for m in tags.get("models", [])]
        self._send_json(200, {"object": "list", "data": data})

    def _handle_openai_chat(self, payload: dict) -> None:
        model = payload.get("model")
        if not model:
            self._send_error_json(400, "missing 'model'")
            return

        ramer = self.server.ramer
        num_ctx = 4096  # OpenAI's request shape carries no context-length option; use the probe default
        num_predict = payload.get("max_tokens") or payload.get("max_completion_tokens")
        client_stream = bool(payload.get("stream", False))

        cached = ramer.placement.get_cached(model, num_ctx=num_ctx)
        if cached:
            num_gpu = cached["recommended"]["num_gpu"]
            source = "ramer-cache"
        else:
            num_gpu = "auto"
            source = f"auto -- no cached placement, run `ramer benchmark {model}` once"
        ramer.log(f"/v1/chat/completions model={model} num_ctx={num_ctx} num_gpu={num_gpu} ({source})")

        chat_id = _openai_id()
        created = int(time.time())
        started = threading.Event()
        role_sent = [False]  # mutable cell -- on_chunk is a closure, not a method, so no `self` to hold this flag on

        def on_chunk(chunk: dict) -> None:
            if not client_stream:
                return  # buffered mode: the final response built below is all that gets sent
            msg = chunk.get("message", {})
            delta = {}
            if not role_sent[0]:
                delta["role"] = "assistant"
                role_sent[0] = True
            if msg.get("content"):
                delta["content"] = msg["content"]
            if msg.get("tool_calls"):
                delta["tool_calls"] = _ollama_tool_calls_to_openai(msg["tool_calls"])
            finish_reason = None
            if chunk.get("done"):
                finish_reason = "tool_calls" if msg.get("tool_calls") else "stop"
            if len(delta) <= 1 and "role" in delta and finish_reason is None:
                return  # nothing but a bare role marker and nothing finished yet -- not worth a chunk on its own
            if not started.is_set():
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                started.set()
            data = _openai_stream_chunk(chat_id, created, model, delta, finish_reason)
            try:
                self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError):
                pass  # client disconnected mid-stream -- nothing left to do for this request

        agent_id = self.headers.get("X-Ramer-Agent", f"{self.client_address[0]}:{self.client_address[1]}")
        priority = int(self.headers.get("X-Ramer-Priority", 0))
        try:
            result = ramer.scheduler.submit(
                agent_id=agent_id, model=model, priority=priority,
                messages=_normalize_openai_messages(payload.get("messages", [])),
                num_gpu=num_gpu, num_ctx=num_ctx, tools=payload.get("tools"), num_predict=num_predict,
                on_chunk=on_chunk,
            )
        except urllib.error.URLError as e:
            if not started.is_set():
                self._send_error_json(502, f"upstream Ollama unreachable: {e}")
            return
        except Exception as e:
            if not started.is_set():
                self._send_error_json(500, str(e))
            return

        if client_stream:
            if not started.is_set():
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError):
                pass
            return

        message = result.get("message", {})
        tool_calls = _ollama_tool_calls_to_openai(message.get("tool_calls"))
        openai_message = {"role": "assistant", "content": message.get("content") or None}
        if tool_calls:
            openai_message["tool_calls"] = tool_calls
        prompt_tokens = result.get("prompt_eval_count", 0)
        completion_tokens = result.get("eval_count", 0)
        self._send_json(200, {
            "id": chat_id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{
                "index": 0, "message": openai_message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })

    def do_POST(self) -> None:
        if self.path in PASSTHROUGH_POST_PATHS:
            self._proxy_passthrough("POST")
            return

        try:
            payload = self._read_json()
        except json.JSONDecodeError:
            self._send_error_json(400, "invalid JSON body")
            return

        if self.path == "/v1/chat/completions":
            self._handle_openai_chat(payload)
            return
        if self.path not in ("/api/chat", "/api/generate"):
            self._send_error_json(404, f"unhandled path {self.path}")
            return

        model = payload.get("model")
        if not model:
            self._send_error_json(400, "missing 'model'")
            return

        ramer = self.server.ramer
        options = payload.get("options") or {}
        num_ctx = options.get("num_ctx", 4096)
        num_predict = options.get("num_predict")
        client_stream = payload.get("stream", True)

        if "num_gpu" in options:
            num_gpu = options["num_gpu"]
            source = "request"
        else:
            cached = ramer.placement.get_cached(model, num_ctx=num_ctx)
            if cached:
                num_gpu = cached["recommended"]["num_gpu"]
                source = "ramer-cache"
            else:
                num_gpu = "auto"
                source = f"auto -- no cached placement, run `ramer benchmark {model} --num-ctx {num_ctx}` once"
        ramer.log(f"{self.path} model={model} num_ctx={num_ctx} num_gpu={num_gpu} ({source})")

        started = threading.Event()

        def on_chunk(chunk: dict) -> None:
            if not client_stream:
                return  # buffered mode: the final `result` below is all that gets sent
            if not started.is_set():
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Connection", "close")
                self.end_headers()
                started.set()
            try:
                self.wfile.write((json.dumps(chunk) + "\n").encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError):
                pass  # client disconnected mid-stream -- nothing left to do for this request

        try:
            if self.path == "/api/chat":
                agent_id = self.headers.get("X-Ramer-Agent", f"{self.client_address[0]}:{self.client_address[1]}")
                priority = int(self.headers.get("X-Ramer-Priority", 0))
                result = ramer.scheduler.submit(
                    agent_id=agent_id, model=model, priority=priority, messages=payload.get("messages", []),
                    num_gpu=num_gpu, num_ctx=num_ctx, tools=payload.get("tools"), num_predict=num_predict,
                    on_chunk=on_chunk,
                )
            else:  # /api/generate -- direct relay, not scheduler-queued (see module docstring)
                extra = {"prompt": payload.get("prompt", "")}
                if "system" in payload:
                    extra["system"] = payload["system"]
                result = _relay("/api/generate", model, extra, num_gpu, num_ctx, num_predict, ramer.upstream, on_chunk=on_chunk)
        except urllib.error.URLError as e:
            if not started.is_set():
                self._send_error_json(502, f"upstream Ollama unreachable: {e}")
            return
        except Exception as e:
            if not started.is_set():
                self._send_error_json(500, str(e))
            return

        if client_stream:
            if not started.is_set():
                # zero chunks arrived (e.g. num_predict=0) -- still owe a response
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Connection", "close")
                self.end_headers()
        else:
            self._send_json(200, result)


class RamerProxyServer:
    """The long-running object the CLI's `serve` command and (eventually)
    the GUI both wrap: one shared RamerScheduler + PlacementEngine behind
    an HTTP listener, so every agent tool pointed at this port shares the
    same residency/priority decisions instead of each independently racing
    Ollama -- the exact failure mode Experiment D measured."""

    def __init__(self, upstream: str = DEFAULT_UPSTREAM, port: int = DEFAULT_PROXY_PORT,
                 on_log: Optional[Callable[[str], None]] = None):
        self.upstream = upstream
        self.port = port
        self.placement = PlacementEngine()
        self.on_log = on_log or (lambda msg: None)
        self.scheduler = RamerScheduler(chat_impl=self._chat_impl)
        self._httpd: Optional[ThreadingHTTPServer] = None

    def log(self, msg: str) -> None:
        self.on_log(msg)

    def status(self) -> dict:
        return dict(
            upstream=self.upstream, resident_model=self.scheduler._resident_model,
            queue_length=len(self.scheduler._queue), **self.scheduler.stats,
        )

    def _chat_impl(self, model, messages, num_gpu, num_ctx, tools=None, num_predict=None, on_chunk=None) -> dict:
        return _relay("/api/chat", model, {"messages": messages, "tools": tools if tools is not None else []},
                       num_gpu, num_ctx, num_predict, self.upstream, on_chunk=on_chunk)

    def serve_forever(self) -> None:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), _Handler)
        self._httpd.ramer = self  # each handler reaches config/scheduler back via self.server.ramer
        self.port = self._httpd.server_address[1]  # resolves an actual port when constructed with port=0
        self.log(f"Ramer proxy listening on http://127.0.0.1:{self.port} -> {self.upstream}")
        try:
            self._httpd.serve_forever()
        finally:
            self._httpd.server_close()

    def start_background(self) -> threading.Thread:
        """For tests and the GUI: runs serve_forever() on a daemon thread and
        blocks just until the listening socket exists, so a caller can start
        making requests immediately after this returns."""
        t = threading.Thread(target=self.serve_forever, daemon=True)
        t.start()
        for _ in range(200):
            if self._httpd is not None:
                break
            time.sleep(0.02)
        return t

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
        self.scheduler.stop()
