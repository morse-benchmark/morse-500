"""Run official coding-harness evaluations on identical cached MORSE-500 frames."""

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.gateway import Gateway
from estimates.usage import Ledger
from lib import run_paths
from lib.budget import Budget, BudgetExceeded
from lib import common as api
from lib import frames as frame_utils


PROMPT = (
    "Answer the question in these video frames, in chronological order. "
    "They are all the visual evidence available. You may write and execute code "
    "in /work to reason about these frames. Do not use external sources. "
    "Return your final answer with any necessary explanation."
)
RELAY_PORT = 8080
# Distinct exit status: the provider/model/harness is unusable, so stop the sweep.
EXIT_UNREACHABLE = 75
# Distinct exit status: the shared USD budget is spent; resume later with a higher one.
EXIT_BUDGET = 76
# Upstream outcomes that will not fix themselves by retrying the next example.
FATAL_HTTP = {"http_401": "invalid API key", "http_403": "key lacks permission",
              "http_404": "model not found for this key"}
FATAL_TRANSPORT = ("ConnectError", "ConnectTimeout", "ProxyError", "UnsupportedProtocol")
# Failure messages that mean an example legitimately hit its configured budget.
BUDGET_MARKERS = ("exhausted request budget", "per-example walltime", "error_max_turns")
MODEL_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/models/{}",
    "anthropic": "https://api.anthropic.com/v1/models/{}",
}
# Base64 image payloads (optionally as data URLs) inside trace events: JPEG, PNG, GIF, WebP.
IMAGE_BASE64 = re.compile(
    r"(?:data:image/[A-Za-z0-9.+-]+;base64,)?"
    r"(?:/9j/|iVBORw0KGgo|R0lGOD|UklGR)[A-Za-z0-9+/\r\n]{64,}={0,2}"
)
STDERR_TRACE_CHARS = 20000


class Unreachable(Exception):
    """Provider, model, or harness is unusable; continuing would waste the job."""


def provider_key(harness):
    provider = "openai" if harness == "codex" else "anthropic"
    name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    key = os.environ.get(name, "").strip()
    if not key:
        raise ValueError(f"Export {name}; subscription login is not used")
    return provider, key


def check_model(provider, model, key, transport=None):
    """Free GET of the model record: confirms network, key, and model access."""
    if provider == "openai":
        headers = {"authorization": f"Bearer {key}"}
    else:
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    url = MODEL_ENDPOINTS[provider].format(model)
    try:
        with httpx.Client(timeout=httpx.Timeout(30, connect=20),
                          transport=transport, trust_env=False) as client:
            response = client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        raise Unreachable(f"{provider} API unreachable from this node: {type(exc).__name__}")
    if response.status_code != 200:
        try:
            error = response.json().get("error", {})
            detail = error.get("code") or error.get("type") or ""
        except ValueError:
            detail = ""
        reason = FATAL_HTTP.get(f"http_{response.status_code}", "unexpected response")
        raise Unreachable(
            f"{provider}/{model}: HTTP {response.status_code} ({reason}) {detail}".strip()
        )
    return response.json().get("id", model)


def fatal_upstream(statuses):
    reasons = set()
    for status in statuses:
        if status in FATAL_HTTP:
            reasons.add(f"{status} {FATAL_HTTP[status]}")
        elif status.removeprefix("gateway_error_") in FATAL_TRANSPORT:
            reasons.add(f"{status} (provider unreachable)")
    return sorted(reasons)


def harness_command(args, frames):
    if args.harness == "claude":
        blocks = [{"type": "text", "text": PROMPT}]
        for path in frames:
            blocks.append({
                "type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                },
            })
        command = [
            "/opt/harness", "--print", "--model", args.model,
            "--input-format", "stream-json", "--output-format", "stream-json",
            "--verbose", "--max-turns", str(args.max_requests),
            "--permission-mode", "bypassPermissions",
            "--tools", "Bash,Read,Write,Edit,Glob,Grep",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--setting-sources", "", "--no-session-persistence",
        ]
        data = json.dumps({
            "type": "user", "message": {"role": "user", "content": blocks},
        }) + "\n"
    else:
        command = [
            "/opt/harness", "--ask-for-approval", "never",
            "-c", 'model_provider="morse"',
            "-c", 'model_providers.morse.name="MORSE gateway"',
            "-c", f'model_providers.morse.base_url="http://127.0.0.1:{args.relay_port}/v1"',
            "-c", 'model_providers.morse.wire_api="responses"',
            "-c", 'model_providers.morse.env_key="CODEX_API_KEY"',
            "-c", "agents.enabled=false",
            "-c", 'web_search="disabled"',
            "-c", "analytics.enabled=false",
            "exec", "--model", args.model, "--json", "--ephemeral",
            "--skip-git-repo-check", "--sandbox", "danger-full-access",
        ]
        for index in range(len(frames)):
            command.extend(["--image", f"/frames/{index:02d}.jpg"])
        command.append("-")
        data = PROMPT
    return command, data


def sandbox_command(binary, scratch, config, socket_dir, code_mode_host=None,
                    socat_binary=None):
    command = [
        "bwrap", "--unshare-all", "--die-with-parent", "--new-session",
        "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
        "--setenv", "LANG", "C.UTF-8",
        "--setenv", "HOME", "/home/agent",
    ]
    for path in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(path).exists():
            command.extend(["--ro-bind", path, path])
    command.extend([
        "--proc", "/proc", "--dev", "/dev",
        "--tmpfs", "/tmp", "--tmpfs", "/home",
        "--dir", "/opt", "--ro-bind", str(binary), "/opt/harness",
    ])
    if code_mode_host is not None:
        # Codex spawns this sibling of its own executable for the GPT-6 `exec` tool.
        command.extend(["--ro-bind", str(code_mode_host), "/opt/codex-code-mode-host"])
    if socat_binary is not None:
        command.extend(["--ro-bind", str(socat_binary), "/opt/morse-socat"])
    command.extend([
        "--ro-bind", str(scratch / "frames"), "/frames",
        "--bind", str(scratch / "work"), "/work",
        "--ro-bind", str(socket_dir), "/gateway",
        "--ro-bind", str(config), "/config.json",
        "--ro-bind", str(Path(__file__).with_name("entry.py")), "/entry.py",
        "--chdir", "/work", "/usr/bin/python3", "/entry.py",
    ])
    return command


def execute(command, timeout, capture=None):
    """Run the sandbox; `capture` (dict) also receives stdout/stderr on timeout."""
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        if capture is not None:
            capture.update(stdout=stdout, stderr=stderr, exit_code=None)
        raise TimeoutError("Harness exceeded per-example walltime; partial usage is in ledger")
    if capture is not None:
        capture.update(stdout=stdout, stderr=stderr, exit_code=process.returncode)
    return process.returncode, stdout, stderr


def frame_hashes(images):
    """SHA-256 of each cached frame's JPEG bytes -> 0-based frame index (/frames/NN.jpg)."""
    return {
        hashlib.sha256(base64.b64decode(image)).hexdigest(): number
        for number, image in enumerate(images)
    }


def image_placeholder(match, hashes):
    payload = match.group(0)
    if payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    payload = re.sub(r"\s", "", payload)
    try:
        data = base64.b64decode(payload, validate=True)
    except ValueError:
        return f"<image omitted base64_chars={len(payload)}>"
    digest = hashlib.sha256(data).hexdigest()
    if digest in hashes:
        return f"<image frame {hashes[digest]}>"
    return f"<image omitted sha256={digest[:16]} bytes={len(data)}>"


def redact_images(value, hashes):
    """Replace base64 images anywhere in a JSON value; exact cached frames become their index."""
    if isinstance(value, str):
        return IMAGE_BASE64.sub(lambda match: image_placeholder(match, hashes), value)
    if isinstance(value, list):
        return [redact_images(item, hashes) for item in value]
    if isinstance(value, dict):
        return {key: redact_images(item, hashes) for key, item in value.items()}
    return value


def trace_events(stdout, hashes):
    events = []
    for line in (stdout or "").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            event = {"type": "non_json_stdout", "text": line}
        events.append(redact_images(event, hashes))
    return events


def write_trace(path, record):
    """Append one attempt; tracing problems never fail or alter the evaluation."""
    try:
        with path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, TypeError, ValueError) as exc:
        print(f"WARNING: could not write trace to {path}: {exc}", file=sys.stderr, flush=True)


def final_answer(harness, stdout):
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    if harness == "claude":
        results = [event for event in events if event.get("type") == "result"]
        if not results or results[-1].get("is_error") or results[-1].get("subtype") != "success":
            subtype = results[-1].get("subtype") if results else "none"
            raise ValueError(f"Claude Code did not emit a successful final result (subtype={subtype})")
        answer = results[-1].get("result", "")
    else:
        if any(event.get("type") in ("error", "turn.failed") for event in events):
            raise ValueError("Codex reported an error/failed turn")
        if not any(event.get("type") == "turn.completed" for event in events):
            raise ValueError("Codex did not complete its agent turn")
        answers = [
            event["item"].get("text", "") for event in events
            if event.get("type") == "item.completed"
            and event.get("item", {}).get("type") == "agent_message"
        ]
        answer = answers[-1] if answers else ""
    if not answer.strip():
        raise ValueError("Harness returned an empty final answer")
    return answer.strip()


def preflight(args):
    args.socat_binary_path = shutil.which(args.socat_binary)
    for tool in ("bwrap",):
        if not shutil.which(tool, path="/usr/bin:/bin"):
            raise FileNotFoundError(f"Required isolation tool is missing: {tool}")
    if not args.socat_binary_path:
        raise FileNotFoundError(f"Required local relay executable is missing: {args.socat_binary}")
    resolved = shutil.which(args.binary)
    if not resolved:
        raise FileNotFoundError(
            f"Official {args.harness} CLI missing: {args.binary}. "
            "Install a native standalone binary; see eval/api/README.md (agent harness setup)."
        )
    binary = Path(resolved).resolve()
    with binary.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError("Use a native Linux ELF harness executable, not an npm/shell wrapper")
    args.code_mode_host_path = None
    if args.harness == "codex":
        # Default: installed beside the (possibly symlinked) codex binary.
        host = Path(args.code_mode_host or Path(resolved).parent / "codex-code-mode-host")
        if not host.is_file():
            raise FileNotFoundError(
                f"Codex code-mode host missing: {host}. GPT-6 models' `exec` tool needs "
                "codex-code-mode-host from the same Codex release; see eval/api/README.md (agent harness setup)."
            )
        host = host.resolve()
        with host.open("rb") as stream:
            if stream.read(4) != b"\x7fELF" or not os.access(host, os.X_OK):
                raise ValueError(f"Codex code-mode host is not an executable ELF: {host}")
        args.code_mode_host_path = host
    with tempfile.TemporaryDirectory(prefix="morse-preflight-") as directory:
        root = Path(directory)
        (root / "frames").mkdir()
        (root / "work").mkdir()
        (root / "socket").mkdir()
        config = root / "config.json"
        config.write_text(json.dumps({
            "command": ["/opt/harness", "--version"], "input": "",
            "relay_port": args.relay_port, "relay_binary": "/opt/morse-socat",
        }))
        code, stdout, stderr = execute(
            sandbox_command(binary, root, config, root / "socket", args.code_mode_host_path,
                            args.socat_binary_path), 30,
        )
        if code:
            raise RuntimeError(f"Isolated harness --version failed: {stderr[-2000:]}")
        if args.code_mode_host_path is not None:
            config.write_text(json.dumps({
                "command": ["/opt/codex-code-mode-host", "--help"], "input": "",
                "relay_port": args.relay_port, "relay_binary": "/opt/morse-socat",
            }))
            code, _, stderr = execute(
                sandbox_command(binary, root, config, root / "socket", args.code_mode_host_path,
                                args.socat_binary_path), 30,
            )
            if code:
                raise RuntimeError(f"Isolated codex-code-mode-host failed to start: {stderr[-2000:]}")
    return binary, stdout.strip()


def make_budget(args):
    if args.budget_usd is None:
        return None
    provider = "openai" if args.harness == "codex" else "anthropic"
    budget = Budget(args.budget_file, args.budget_usd, provider, args.model)
    settled, open_usd = budget.spent()
    print(f"USD budget: ${settled + open_usd:.2f} committed (${open_usd:.2f} reserved) "
          f"of ${budget.limit:.2f}; ledger={budget.path}", flush=True)
    # Smallest possible next request: no input, full output cap.
    budget.require(budget.cost(0, args.max_output_tokens))
    return budget


def run(args):
    binary, version = preflight(args)
    print(f"Official harness: {version}; binary={binary}", flush=True)
    if args.check_model:
        provider, key = provider_key(args.harness)
        resolved = check_model(provider, args.model, key)
        print(f"Model reachable: {provider}/{resolved}", flush=True)
    if args.preflight_only:
        make_budget(args)
        return
    provider, key = provider_key(args.harness)
    rows = api.read_csv(args.input)
    api.validate_rows(rows)
    rows = rows[:args.limit] if args.limit else rows
    if not args.limit and len(rows) != 500:
        raise ValueError("Full benchmark must contain 500 examples")
    for row in rows:
        frame_utils.load_cached_frames(args, row["video"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": "coding-agent-v1", "harness": args.harness,
        "model": args.model, "version": version,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "cache": str(args.frame_cache.resolve()), "sampling": frame_utils.cache_policy(args),
        "sources": [frame_utils.source_identity(args.video_root / row["video"]) for row in rows],
        "max_requests": args.max_requests, "seconds": args.seconds,
        "max_output_tokens": args.max_output_tokens, "limit": args.limit,
        "prompt": PROMPT,
    }
    if args.code_mode_host_path is not None:
        config["code_mode_host_sha256"] = hashlib.sha256(
            args.code_mode_host_path.read_bytes()).hexdigest()
    metadata = args.output.with_suffix(".meta.json")
    if metadata.exists():
        if json.loads(metadata.read_text()) != config:
            raise ValueError("Harness resume configuration changed; choose a new output path")
    elif args.output.exists():
        raise ValueError("Existing output has no metadata; choose a new output path")
    else:
        metadata.write_text(json.dumps(config, indent=2) + "\n")
    existing = api.read_csv(args.output) if args.output.exists() else []
    results = {row["video"]: row for row in existing}
    if len(results) != len(existing) or not set(results).issubset({row["video"] for row in rows}):
        raise ValueError("Checkpoint contains duplicate/unexpected examples")
    ledger = Ledger(args.output, provider, args.model, "agent", args.harness)
    budget = make_budget(args)
    traces = None if args.no_traces else args.output.with_suffix(".traces.jsonl")
    limiter = api.RateLimiter(args.rpm)
    fields = api.FIELDS + ["harness", "requests", "denied_requests", "status"]
    failures = consecutive = 0
    abort = budget_stop = None
    for index, row in enumerate(rows, 1):
        if api.successful(results.get(row["video"], {}).get("prediction", "")):
            continue
        if budget is not None:
            budget.require(budget.cost(0, args.max_output_tokens))
        result = {field: "" for field in fields}
        result.update({field: row[field] for field in ("video", "ground_truth", "question_text")})
        result.update(idx=index, harness=args.harness)
        capture, hashes = {}, {}
        attempt_started = time.time()
        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="morse-agent-") as directory:
                root = Path(directory)
                (root / "frames").mkdir()
                (root / "work").mkdir()
                (root / "socket").mkdir()
                images = frame_utils.load_cached_frames(args, row["video"])
                hashes = frame_hashes(images)
                frames = []
                for number, image in enumerate(images):
                    frame = root / "frames" / f"{number:02d}.jpg"
                    frame.write_bytes(base64.b64decode(image, validate=True))
                    frames.append(frame)
                result["frame_count"] = len(frames)
                command, data = harness_command(args, frames)
                config_path = root / "config.json"
                base_url = f"http://127.0.0.1:{args.relay_port}"
                environment = {
                    "ANTHROPIC_BASE_URL": base_url,
                    "ANTHROPIC_API_KEY": "morse-gateway-placeholder",
                    "CODEX_API_KEY": "morse-gateway-placeholder",
                    "CODEX_HOME": "/home/agent/.codex",
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "DISABLE_AUTOUPDATER": "1",
                }
                config_path.write_text(json.dumps({
                    "command": command, "input": data,
                    "relay_port": args.relay_port,
                    "relay_binary": "/opt/morse-socat",
                    "environment": environment,
                    "directories": ["/home/agent/.codex", "/home/agent/.claude"],
                }))
                gateway = Gateway(
                    root / "socket" / "api.sock", provider, args.model, key,
                    ledger, limiter, args.max_requests, args.seconds,
                    args.max_output_tokens,
                    hashlib.sha256(row["video"].encode()).hexdigest(),
                    budget=budget,
                )
                try:
                    with gateway.running():
                        code, stdout, stderr = execute(
                            sandbox_command(binary, root, config_path, root / "socket",
                                            args.code_mode_host_path, args.socat_binary_path),
                            args.seconds, capture,
                        )
                finally:
                    result.update(requests=gateway.requests, denied_requests=gateway.denied)
                    if gateway.usage_known:
                        result.update(input_tokens=gateway.input_tokens,
                                      output_tokens=gateway.output_tokens)
                    # Checked even after a harness crash/timeout, which a bad key often causes.
                    fatal = fatal_upstream(gateway.statuses)
                    if fatal:
                        raise Unreachable(f"{provider}/{args.model}: {'; '.join(fatal)}")
                    if gateway.usd_denied:
                        # Interrupted example: failed now, retried on resume.
                        raise BudgetExceeded(gateway.usd_error or
                                             f"USD budget of ${budget.limit:.2f} reached mid-example")
                if code:
                    raise RuntimeError(
                        f"Harness exit {code}; diagnostic: {stderr[-2000:]}"
                    )
                if gateway.denied > gateway.budget_denied + gateway.usd_denied:
                    raise ValueError("Harness violated gateway policy")
                if gateway.budget_denied:
                    raise ValueError("Harness exhausted request budget (per-example request/time cap)")
                if gateway.errors:
                    raise ValueError("Gateway transport/accounting error; inspect usage ledger")
                if not gateway.requests:
                    raise ValueError("Harness made no inference requests through the gateway")
                result["prediction"] = final_answer(args.harness, stdout)
                result["status"] = "success"
        except Unreachable as exc:
            print(f"ERROR example {index}: {exc}", file=sys.stderr, flush=True)
            result.update(prediction=f"ERROR: {exc}", status="failed")
            failures += 1
            abort = str(exc)
        except BudgetExceeded as exc:
            print(f"BUDGET example {index}: {exc}", file=sys.stderr, flush=True)
            result.update(prediction=f"ERROR: {exc}", status="failed")
            budget_stop = str(exc)
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            # The real provider key never enters the harness or its output.
            print(f"ERROR example {index}: {exc}", file=sys.stderr, flush=True)
            result.update(prediction=f"ERROR: {exc}", status="failed")
            failures += 1
            if any(marker in str(exc) for marker in BUDGET_MARKERS):
                consecutive = 0  # The harness reached the model; the cap is a valid outcome.
            else:
                consecutive += 1
                if args.max_consecutive_failures and consecutive >= args.max_consecutive_failures:
                    abort = f"{consecutive} consecutive non-budget failures; last: {str(exc)[:500]}"
        else:
            consecutive = 0
        result["elapsed_seconds"] = f"{time.monotonic() - started:.3f}"
        if traces is not None:
            write_trace(traces, {
                "idx": index, "video": row["video"], "harness": args.harness,
                "model": args.model,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(attempt_started)),
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "status": result["status"],
                "error": result["prediction"] if result["status"] == "failed" else None,
                "exit_code": capture.get("exit_code"),
                "requests": result["requests"], "denied_requests": result["denied_requests"],
                "events": trace_events(capture.get("stdout"), hashes),
                "stderr_tail": redact_images(
                    (capture.get("stderr") or "")[-STDERR_TRACE_CHARS:], hashes),
            })
        results[row["video"]] = result
        api.save_csv(
            args.output, [results[r["video"]] for r in rows if r["video"] in results], fields,
        )
        print(f"Checkpoint {index}/{len(rows)}: {result['status']}", flush=True)
        if budget_stop:
            raise BudgetExceeded(budget_stop)
        if abort:
            print(f"ABORT {args.harness}/{args.model}: {abort}", file=sys.stderr, flush=True)
            raise SystemExit(EXIT_UNREACHABLE)
    if failures:
        results[row["video"]] = result
        api.save_csv(
            args.output, [results[r["video"]] for r in rows if r["video"] in results], fields,
        )
        print(f"Checkpoint {index}/{len(rows)}: {result['status']}", flush=True)
    if failures:
        raise RuntimeError(f"{failures} failed examples; checkpoints and usage retained")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, choices=["claude", "codex"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", help="Path to official native standalone executable")
    parser.add_argument("--code-mode-host", default=os.environ.get("CODEX_CODE_MODE_HOST") or None,
                        help="Codex only: codex-code-mode-host executable "
                             "(default: $CODEX_CODE_MODE_HOST, else beside --binary)")
    parser.add_argument("--input", type=Path, default=api.DATA / "test.csv")
    parser.add_argument("--video-root", type=Path, default=api.DATA / "test_sz512")
    parser.add_argument("--frame-cache", type=Path, default=api.REPO / "eval/api/frame_cache/fps4_max32_sz512")
    parser.add_argument("--output", type=Path)
    run_paths.add_arguments(parser)
    parser.add_argument("--fps", type=float, default=4)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--max-requests", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--rpm", type=float, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--check-model", action="store_true",
                        help="Free model-record GET with the real key; exit 75 if unreachable")
    parser.add_argument("--max-consecutive-failures", type=int, default=0,
                        help="Exit 75 after this many consecutive non-budget failures (0 = never)")
    parser.add_argument("--budget-usd", type=float,
                        help="Cumulative standard-rate USD cap shared via --budget-file; "
                             f"exit {EXIT_BUDGET} when reached (default: no cap)")
    parser.add_argument("--budget-file", type=Path,
                        help="Shared budget ledger (default: RUN_DIR/run_usage/budget.jsonl)")
    parser.add_argument("--no-traces", action="store_true",
                        help="Do not append harness event streams to OUTPUT.traces.jsonl")
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT,
                        help="Local sandbox-to-gateway relay port")
    parser.add_argument("--socat-binary", default=os.environ.get("SOCAT_BINARY", "socat"),
                        help="Path or command name for the local TCP-to-Unix-socket relay")
    args = parser.parse_args(argv)
    args.binary = args.binary or args.harness
    for name in ("fps", "max_frames", "max_side", "max_requests",
                 "seconds", "max_output_tokens", "rpm", "limit"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.max_consecutive_failures < 0:
        parser.error("--max-consecutive-failures must be >= 0")
    if not 1 <= args.relay_port <= 65535:
        parser.error("--relay-port must be between 1 and 65535")
    if args.budget_usd is not None:
        if not math.isfinite(args.budget_usd) or args.budget_usd <= 0:
            parser.error("--budget-usd must be finite and positive")
        if args.budget_file is None:
            if not args.output and args.preflight_only:
                parser.error("--budget-file is required with --budget-usd and --preflight-only")
            if args.output:
                args.budget_file = run_paths.usage_path(args.output).parent / "budget.jsonl"
    return args


if __name__ == "__main__":
    try:
        args = parse_args()
        if not args.preflight_only:
            run_paths.initialize_evaluator(args, f"agent-{args.harness}")
        run(args)
    except Unreachable as exc:
        print(f"UNREACHABLE: {exc}", file=sys.stderr, flush=True)
        sys.exit(EXIT_UNREACHABLE)
    except BudgetExceeded as exc:
        print(f"BUDGET REACHED: {exc}", file=sys.stderr, flush=True)
        sys.exit(EXIT_BUDGET)
