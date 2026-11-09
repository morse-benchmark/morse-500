"""Credential-hiding, model-pinned Unix-socket gateway for isolated harnesses."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
import json
import socketserver
import threading
import time
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx

from estimates.usage import token_totals


class UsageObject:
    def __init__(self, value):
        self.value = value

    def model_dump(self, mode):
        return self.value


def collect_usage(event, provider, state):
    if provider == "openai":
        response = event.get("response", event)
        if response.get("usage") is not None:
            state.update(
                usage=response["usage"], id=response.get("id"),
                model=response.get("model"), status=response.get("status", "received"),
            )
    elif event.get("type") == "message_start":
        message = event["message"]
        state.update(
            usage=dict(message.get("usage", {})),
            id=message.get("id"), model=message.get("model"), status="streaming",
        )
    elif event.get("type") == "message_delta":
        state.setdefault("usage", {}).update(event.get("usage", {}))
        state["status"] = event.get("delta", {}).get("stop_reason") or "streaming"
    elif event.get("type") == "message" and event.get("usage") is not None:
        state.update(usage=event["usage"], id=event.get("id"),
                     model=event.get("model"), status=event.get("stop_reason"))


def validate_request(provider, model, path, body):
    expected = "/v1/responses" if provider == "openai" else "/v1/messages"
    parsed = urlsplit(path)
    allowed_queries = ("", "beta=true") if provider == "anthropic" else ("",)
    if parsed.path != expected or parsed.query not in allowed_queries or parsed.netloc:
        raise ValueError("Only the configured inference endpoint is permitted")
    if body.get("model") != model:
        raise ValueError("Model switching/fallback is not permitted")
    for tool in body.get("tools", []):
        # Built-in server-side web/search/MCP tools could bypass network isolation.
        allowed = (None, "custom") if provider == "anthropic" else ("function", "custom")
        if tool.get("type") not in allowed:
            raise ValueError("Only local harness tools are permitted")
    if body.get("background") or body.get("previous_response_id"):
        raise ValueError("Background and previous-response access are not permitted")

    def check_images(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "image_url" and isinstance(item, str) and not item.startswith("data:image/"):
                    raise ValueError("Remote image URLs are not permitted")
                if key == "source" and isinstance(item, dict) and item.get("type") == "url":
                    raise ValueError("Remote image URLs are not permitted")
                if key in ("file_id", "container", "vector_store_ids"):
                    raise ValueError("External API resources are not permitted")
                check_images(item)
        elif isinstance(value, list):
            for item in value:
                check_images(item)
    check_images(body)


class Gateway:
    def __init__(self, path, provider, model, key, ledger, limiter,
                 max_requests, timeout, max_output_tokens, example_id, transport=None,
                 budget=None):
        self.provider = provider
        self.model = model
        self.key = key
        self.ledger = ledger
        self.limiter = limiter
        self.max_requests = max_requests
        self.deadline = time.monotonic() + timeout
        self.max_output_tokens = max_output_tokens
        self.example_id = example_id
        self.requests = 0
        self.denied = 0
        self.budget_denied = 0
        # Requests refused because the shared USD budget (budget.Budget) is exhausted/unusable.
        self.budget = budget
        self.usd_denied = 0
        self.usd_error = None
        self.errors = 0
        self.statuses = []
        # Per-example sums over requests whose usage was reported (as in usage.py).
        self.input_tokens = 0
        self.output_tokens = 0
        self.usage_known = 0
        self.lock = threading.Lock()
        self.client = httpx.Client(
            timeout=httpx.Timeout(60, connect=20),
            transport=transport, trust_env=False,
        )
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def deny(self, status, message):
                with gateway.lock:
                    gateway.denied += 1
                data = json.dumps({"error": {"type": "morse_gateway_error", "message": message}}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.deny(403, "Only inference POST requests are allowed")

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 64 * 1024 * 1024:
                        raise ValueError("Invalid or oversized request body")
                    body = json.loads(self.rfile.read(length))
                    validate_request(gateway.provider, gateway.model, self.path, body)
                except (ValueError, TypeError, AttributeError) as exc:
                    self.deny(403, str(exc))
                    return
                with gateway.lock:
                    if gateway.requests >= gateway.max_requests or time.monotonic() >= gateway.deadline:
                        exhausted = True
                    else:
                        exhausted = False
                        gateway.requests += 1
                        attempt = gateway.requests
                if exhausted:
                    with gateway.lock:
                        gateway.budget_denied += 1
                    self.deny(429, "Per-example inference-request/time budget exhausted")
                    return
                gateway.forward(self, body, attempt)

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = False
            # BaseHTTPRequestHandler expects an IP-style client address.
            def get_request(self):
                request, _ = super().get_request()
                request.settimeout(65)
                return request, ("local", 0)

        self.server = Server(str(path), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def forward(self, handler, body, attempt):
        self.limiter.wait()
        started = time.monotonic()
        state = {}
        headers = {"content-type": "application/json"}
        if self.provider == "openai":
            url = "https://api.openai.com/v1/responses"
            headers["authorization"] = f"Bearer {self.key}"
            body["store"] = False
            field = "max_output_tokens"
        else:
            url = "https://api.anthropic.com/v1/messages"
            if urlsplit(handler.path).query:
                url += "?beta=true"
            headers["x-api-key"] = self.key
            headers["anthropic-version"] = "2023-06-01"
            if handler.headers.get("anthropic-beta"):
                headers["anthropic-beta"] = handler.headers["anthropic-beta"]
            field = "max_tokens"
        for name in ("user-agent", "x-app"):
            if handler.headers.get(name):
                headers[name] = handler.headers[name]
        requested = body.get(field, self.max_output_tokens)
        if not isinstance(requested, int) or requested <= 0:
            handler.deny(400, "Invalid output-token budget")
            return
        body[field] = min(requested, self.max_output_tokens)
        reservation = reserved_usd = None
        if self.budget is not None:
            try:
                reserved_usd = self.budget.worst_case(body, body[field])
                reservation = self.budget.reserve(reserved_usd, self.example_id)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                # Fail closed: never spend without a recorded reservation.
                with self.lock:
                    self.usd_denied += 1
                    self.usd_error = f"budget ledger unavailable: {type(exc).__name__}: {exc}"
                handler.deny(503, "MORSE USD budget ledger unavailable")
                return
            if reservation is None:
                with self.lock:
                    self.usd_denied += 1
                handler.deny(400, "MORSE USD budget reached; no further inference requests")
                return
        try:
            if time.monotonic() >= self.deadline:
                with self.lock:
                    self.budget_denied += 1
                handler.deny(429, "Per-example time budget exhausted")
                return
            with self.client.stream("POST", url, headers=headers, json=body) as response:
                state["status"] = f"http_{response.status_code}"
                state["accepted"] = response.status_code < 400
                handler.send_response(response.status_code)
                content_type = response.headers.get("content-type", "application/json")
                handler.send_header("content-type", content_type)
                handler.send_header("connection", "close")
                handler.end_headers()
                pending = b""
                for chunk in response.iter_bytes():
                    if time.monotonic() >= self.deadline:
                        state["status"] = "timeout"
                        break
                    if "text/event-stream" in content_type:
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            if line.startswith(b"data:"):
                                data = line[5:].strip()
                                if data and data != b"[DONE]":
                                    collect_usage(json.loads(data), self.provider, state)
                    else:
                        pending += chunk
                    handler.wfile.write(chunk)
                    handler.wfile.flush()
                if "text/event-stream" not in content_type and pending and response.is_success:
                    collect_usage(json.loads(pending), self.provider, state)
        except (httpx.HTTPError, OSError, ValueError, TypeError, KeyError) as exc:
            # Never log raw provider responses or the injected key.
            state["status"] = f"gateway_error_{type(exc).__name__}"
            with self.lock:
                self.errors += 1
            print(f"Gateway failure: {type(exc).__name__}", file=sys.stderr, flush=True)
        finally:
            with self.lock:
                self.statuses.append(state.get("status", "unknown"))
            if reservation is not None:
                try:
                    self.budget.settle(
                        reservation, reserved_usd, state.get("usage"),
                        str(state.get("status") or "unknown"), state.get("accepted", False),
                    )
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    # The reservation stays counted at its worst case.
                    with self.lock:
                        self.errors += 1
                    print(f"Gateway budget settlement failure: {type(exc).__name__}",
                          file=sys.stderr, flush=True)
            response = None
            if state.get("usage") is not None:
                try:
                    totals = token_totals({"provider": self.provider, "usage": state["usage"]})
                except (ValueError, TypeError, AttributeError):
                    totals = None
                if totals is not None:
                    with self.lock:
                        self.input_tokens += totals[0]
                        self.output_tokens += totals[1]
                        self.usage_known += 1
                response = SimpleNamespace(
                    id=state.get("id"), model=state.get("model"),
                    usage=UsageObject(state["usage"]),
                )
            try:
                self.ledger.record(
                    response, time.monotonic() - started,
                    state.get("status", "unknown"), attempt, self.example_id,
                )
            except OSError as exc:
                with self.lock:
                    self.errors += 1
                print(f"Gateway accounting failure: {type(exc).__name__}",
                      file=sys.stderr, flush=True)

    @contextmanager
    def running(self):
        self.thread.start()
        try:
            yield self
        finally:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join()
            self.client.close()
