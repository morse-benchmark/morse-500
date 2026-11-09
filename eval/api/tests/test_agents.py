import base64
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import Mock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents import run_agents_eval as agents
from agents.gateway import Gateway, collect_usage, validate_request
from estimates import usage
from tests.session_helper import session_env


class GatewayTests(unittest.TestCase):
    def test_pinning_and_no_remote_tools(self):
        validate_request("anthropic", "model", "/v1/messages?beta=true", {
            "model": "model", "tools": [{"name": "Bash", "input_schema": {}}],
        })
        validate_request("openai", "model", "/v1/responses", {
            "model": "model", "tools": [{"type": "function", "name": "shell"}],
        })
        for path, body in [
            ("/v1/files", {"model": "model"}),
            ("/v1/responses", {"model": "different"}),
            ("/v1/responses", {"model": "model", "tools": [{"type": "web_search"}]}),
            ("/v1/responses", {"model": "model", "previous_response_id": "other-session"}),
            ("/v1/responses", {"model": "model", "input": [{"image_url": "https://example.com/x.jpg"}]}),
            ("/v1/responses", {"model": "model", "input": [{"file_id": "file_private"}]}),
        ]:
            with self.assertRaises(ValueError):
                validate_request("openai", "model", path, body)

    def test_usage_stream_normalization(self):
        state = {}
        collect_usage({
            "type": "message_start", "message": {
                "id": "msg1", "model": "model",
                "usage": {"input_tokens": 20, "output_tokens": 0, "cache_read_input_tokens": 10},
            },
        }, "anthropic", state)
        collect_usage({
            "type": "message_delta", "usage": {"output_tokens": 30},
            "delta": {"stop_reason": "end_turn"},
        }, "anthropic", state)
        self.assertEqual(state["usage"]["input_tokens"], 20)
        self.assertEqual(state["usage"]["output_tokens"], 30)
        self.assertEqual(state["status"], "end_turn")

    def test_gateway_injects_key_limits_requests_and_records_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seen = []
            def upstream(request):
                seen.append(request)
                return httpx.Response(200, json={
                    "id": "resp1", "model": "model", "status": "completed",
                    "usage": {"input_tokens": 20, "output_tokens": 5},
                })
            ledger = usage.Ledger(root / "pred.csv", "openai", "model", "agent", "codex")
            gateway = Gateway(
                root / "api.sock", "openai", "model", "host-secret",
                ledger, Mock(), 1, 30, 100, "hashed",
                transport=httpx.MockTransport(upstream),
            )
            with gateway.running(), httpx.Client(
                transport=httpx.HTTPTransport(uds=str(root / "api.sock")),
                base_url="http://gateway",
            ) as client:
                response = client.post("/v1/responses", json={
                    "model": "model", "max_output_tokens": 1000,
                })
                self.assertEqual(response.status_code, 200)
                self.assertEqual(client.post("/v1/responses", json={"model": "model"}).status_code, 429)
                self.assertEqual(client.get("/v1/files").status_code, 403)
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0].headers["authorization"], "Bearer host-secret")
            self.assertEqual(json.loads(seen[0].content)["max_output_tokens"], 100)
            events = [json.loads(line) for line in ledger.path.read_text().splitlines()]
            self.assertEqual(events[0]["usage"]["input_tokens"], 20)
            self.assertNotIn("host-secret", ledger.path.read_text())
            self.assertEqual(gateway.denied, 2)
            self.assertEqual((gateway.input_tokens, gateway.output_tokens), (20, 5))

    def test_sse_forwarding_preserves_messages_and_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [
                {"type": "message_start", "message": {
                    "id": "msg", "model": "model",
                    "usage": {"input_tokens": 10, "output_tokens": 0,
                              "cache_read_input_tokens": 3},
                }},
                {"type": "message_delta", "usage": {"output_tokens": 4},
                 "delta": {"stop_reason": "end_turn"}},
            ]
            data = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            ledger = usage.Ledger(root / "pred.csv", "anthropic", "model", "agent", "claude")
            gateway = Gateway(
                root / "api.sock", "anthropic", "model", "host-secret",
                ledger, Mock(), 2, 30, 100, "hashed",
                transport=httpx.MockTransport(lambda request: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, content=data,
                )),
            )
            with gateway.running(), httpx.Client(
                transport=httpx.HTTPTransport(uds=str(root / "api.sock")),
                base_url="http://gateway",
            ) as client:
                response = client.post("/v1/messages", json={"model": "model", "stream": True})
                self.assertEqual(response.text, data)
            event = json.loads(ledger.path.read_text())
            self.assertEqual(event["usage"]["input_tokens"], 10)
            self.assertEqual(event["usage"]["output_tokens"], 4)
            # Matches usage.py: Anthropic cache reads count as input.
            self.assertEqual((gateway.input_tokens, gateway.output_tokens), (13, 4))


class HarnessTests(unittest.TestCase):
    def test_unified_runner_maps_providers_to_official_harnesses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "python"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import os,sys,json\n"
                "with open(os.environ['CALL_LOG'],'a') as f:\n"
                " f.write(json.dumps(sys.argv[1:])+'\\n')\n"
                "if sys.argv[1].endswith('lib/run_paths.py'):\n"
                " os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
            )
            fake.chmod(0o700)
            bindir = root / "bin"
            bindir.mkdir()
            for name in ("bwrap", "socat"):
                tool = bindir / name
                tool.write_text("#!/usr/bin/env bash\nexit 0\n")
                tool.chmod(0o700)
            executable = root / "harness"
            executable.write_text("#!/usr/bin/env bash\nexit 0\n")
            executable.chmod(0o700)
            code_host = root / "codex-code-mode-host"
            code_host.write_text("#!/usr/bin/env bash\nexit 0\n")
            code_host.chmod(0o700)
            for provider, harness, count in (("openai", "codex", 5), ("anthropic", "claude", 3)):
                log = root / f"{provider}.jsonl"
                env = session_env(
                    root, PYTHON=str(fake), CALL_LOG=str(log),
                    REPO=str(root), OPENAI_API_KEY="test",
                    DATA_CSV=str(agents.api.DATA / "test.csv"), VIDEO_ROOT=str(root),
                    ANTHROPIC_API_KEY="test", OUTPUT_DIR=str(root / f"eval/results/{provider}"),
                    FRAME_CACHE=str(root / "frames"), LIMIT="1",
                    PATH=f"{bindir}:/usr/bin:/bin",
                )
                env["CODEX_BINARY"] = str(executable)
                env["CLAUDE_BINARY"] = str(executable)
                env["CODEX_CODE_MODE_HOST"] = str(code_host)
                subprocess.run([
                    "bash", str(agents.api.REPO / "eval/api/scripts/run_local.sh"),
                    "agent", provider,
                ], env=env, check=True, capture_output=True, text=True)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                preflights = [call for call in calls if "--preflight-only" in call]
                self.assertEqual(len(preflights), count)
                for call in preflights:
                    self.assertEqual(Path(call[0]), Path(agents.__file__))
                    self.assertTrue(Path(call[0]).is_file())
                    self.assertEqual(call[call.index("--harness") + 1], harness)
                    self.assertIn("--check-model", call)
                runs = [call for call in calls if "--preflight-only" not in call
                        and call[0].endswith("agents/run_agents_eval.py")]
                self.assertEqual(len(runs), count)
                self.assertTrue(all(call[call.index("--harness") + 1] == harness
                                    for call in runs))
                self.assertTrue(all(call[call.index("--limit") + 1] == "1"
                                    for call in runs))
                self.assertTrue(all(call[call.index("--model") + 1] for call in preflights))

    def test_agent_defaults_live_in_shared_run_configuration(self):
        text = (agents.api.REPO / "eval/api/scripts/config.sh").read_text()
        self.assertIn("OPENAI_AGENT_MODELS=", text)
        self.assertIn("ANTHROPIC_AGENT_MODELS=", text)

    def test_trace_redaction_labels_cached_frames(self):
        frames = [b"\xff\xd8\xff\xe0" + bytes([n]) * 120 for n in range(3)]
        images = [base64.b64encode(frame).decode() for frame in frames]
        hashes = agents.frame_hashes(images)
        other = base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(200)).decode()
        wrapped = "\n".join(images[1][i:i + 76] for i in range(0, len(images[1]), 76))
        stdout = "\n".join([
            json.dumps({"type": "user", "message": {"content": [{"type": "tool_result", "content": [
                {"type": "image", "source": {"type": "base64", "data": images[2]}}]}]},
                "tool_use_result": {"file": {"base64": images[0]}}}),
            json.dumps({"type": "item.completed", "item": {"text":
                f"see data:image/jpeg;base64,{images[1]} and {other} done"}}),
            json.dumps({"wrapped": wrapped}),
            "plain text line",
        ])
        events = agents.trace_events(stdout, hashes)
        dumped = json.dumps(events)
        for image in images:
            self.assertNotIn(image[:40], dumped)
        self.assertNotIn(other[:40], dumped)
        self.assertEqual(
            events[0]["message"]["content"][0]["content"][0]["source"]["data"], "<image frame 2>")
        self.assertEqual(events[0]["tool_use_result"]["file"]["base64"], "<image frame 0>")
        text = events[1]["item"]["text"]
        self.assertTrue(text.startswith("see <image frame 1> and <image omitted sha256="), text)
        self.assertTrue(text.endswith(" done"))
        self.assertEqual(events[2]["wrapped"], "<image frame 1>")
        self.assertEqual(events[3], {"type": "non_json_stdout", "text": "plain text line"})
        self.assertEqual(agents.redact_images("short /9j/abc", hashes), "short /9j/abc")

    def test_walltime_keeps_partial_output_for_traces(self):
        capture = {}
        with self.assertRaises(TimeoutError):
            agents.execute(["/usr/bin/python3", "-c",
                            "import time;print('partial',flush=True);time.sleep(20)"], 1, capture)
        self.assertEqual(capture["stdout"].strip(), "partial")
        self.assertIsNone(capture["exit_code"])

    def test_walltime_limit_terminates_process_group(self):
        with self.assertRaises(TimeoutError):
            agents.execute(["/usr/bin/python3", "-c", "import time;time.sleep(20)"], .1)

    def test_final_events_require_completion(self):
        self.assertEqual(agents.final_answer("claude", json.dumps({
            "type": "result", "subtype": "success", "is_error": False, "result": "answer",
        })), "answer")
        for text in [
            '{"type":"result","subtype":"error_max_turns","is_error":true}',
            '{"type":"assistant"}',
        ]:
            with self.assertRaises(ValueError):
                agents.final_answer("claude", text)
        codex = '\n'.join(json.dumps(event) for event in [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}},
            {"type": "turn.completed", "usage": {}},
        ])
        self.assertEqual(agents.final_answer("codex", codex), "answer")
        with self.assertRaises(ValueError):
            agents.final_answer("codex", codex + '\n{"type":"turn.failed"}')

    def test_harness_inputs_share_same_jpeg_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.jpg"
            frame.write_bytes(b"original-jpeg")
            args = agents.parse_args(["--harness", "claude", "--model", "model", "--preflight-only"])
            command, data = agents.harness_command(args, [frame])
            image = json.loads(data)["message"]["content"][1]["source"]["data"]
            self.assertEqual(base64.b64decode(image), b"original-jpeg")
            self.assertIn("--max-turns", command)
            args.harness = "codex"
            command, data = agents.harness_command(args, [frame])
            self.assertIn("/frames/00.jpg", command)
            self.assertNotIn("original", data)

    def test_missing_harness_fails_explicitly(self):
        args = agents.parse_args([
            "--harness", "claude", "--model", "model", "--preflight-only",
            "--binary", "/nonexistent/morse-claude",
        ])
        with self.assertRaisesRegex(FileNotFoundError, "CLI missing"):
            agents.preflight(args)

    def test_codex_requires_code_mode_host_beside_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "codex").symlink_to("/usr/bin/true")
            args = agents.parse_args([
                "--harness", "codex", "--model", "m", "--preflight-only",
                "--binary", str(root / "codex"),
            ])
            with self.assertRaisesRegex(FileNotFoundError, "code-mode host missing"):
                agents.preflight(args)
            (root / "codex-code-mode-host").symlink_to("/usr/bin/true")
            agents.preflight(args)
            self.assertEqual(args.code_mode_host_path, Path("/usr/bin/true").resolve())
            command = agents.sandbox_command(
                Path("/usr/bin/true"), root, root / "c.json", root, args.code_mode_host_path)
            index = command.index("/opt/codex-code-mode-host")
            self.assertEqual(command[index - 2:index], ["--ro-bind", str(args.code_mode_host_path)])

    def test_real_namespace_hides_host_files_keys_and_network(self):
        if not shutil.which("bwrap") or not shutil.which("socat"):
            self.skipTest("OS namespace tools unavailable")
        with tempfile.TemporaryDirectory(prefix="morse-isolation-test-") as directory:
            root = Path(directory)
            (root / "frames").mkdir()
            (root / "work").mkdir()
            (root / "socket").mkdir()
            secret = root / "ground_truth.csv"
            secret.write_text("must not be readable")
            config = root / "config.json"
            script = (
                "import os,socket,pathlib;"
                f"assert not pathlib.Path({str(secret)!r}).exists();"
                "assert 'OPENAI_API_KEY' not in os.environ;"
                "assert os.environ['ANTHROPIC_API_KEY']=='morse-gateway-placeholder';"
                "pathlib.Path('/work/result').write_text('ok');"
                "s=socket.socket();s.settimeout(0.1)\n"
                "try:\n s.connect(('1.1.1.1',443));raise AssertionError('network escaped')\n"
                "except OSError:\n pass\n"
                "print('isolated-ok')"
            )
            config.write_text(json.dumps({
                "command": ["/opt/harness", "-c", script], "input": "",
                "relay_port": 8080,
                "environment": {
                    "ANTHROPIC_API_KEY": "morse-gateway-placeholder",
                },
            }))
            with patch.dict("os.environ", {"OPENAI_API_KEY": "host-secret"}):
                code, stdout, stderr = agents.execute(agents.sandbox_command(
                    Path("/usr/bin/python3").resolve(), root, config, root / "socket",
                ), 20)
            self.assertEqual(code, 0, stderr)
            self.assertIn("isolated-ok", stdout)
            self.assertEqual((root / "work/result").read_text(), "ok")
            self.assertNotIn("host-secret", stdout + stderr)

    def test_end_to_end_isolated_mock_harness_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory(prefix="morse-agent-integration-") as directory:
            root = Path(directory)
            (root / "clip.mp4").touch()
            input_path = root / "test.csv"
            agents.api.save_csv(input_path, [{
                "video": "clip.mp4", "ground_truth": "secret-label",
                "question_text": "private-metadata",
            }], ["video", "ground_truth", "question_text"])
            args = agents.parse_args([
                "--harness", "claude", "--model", "model",
                "--binary", "/usr/bin/python3", "--input", str(input_path),
                "--video-root", str(root), "--limit", "1", "--rpm", "600",
                "--output", str(root / "pred.csv"),
            ])
            frame = base64.b64encode(b"\xff\xd8\xff\xe0" + bytes(100)).decode()
            script = (
                "import urllib.request,json,os;"
                "assert os.path.isdir(os.environ['CODEX_HOME']), 'CODEX_HOME missing';"
                "assert os.path.isdir(os.environ['HOME'] + '/.claude'), '~/.claude missing';"
                "req=urllib.request.Request('http://127.0.0.1:8080/v1/messages',"
                "data=json.dumps({'model':'model','max_tokens':50}).encode(),"
                "headers={'content-type':'application/json'});"
                "response=json.load(urllib.request.urlopen(req));"
                "assert response['content'][0]['text']=='answer';"
                f"print(json.dumps({{'type':'user','image':{frame!r}}}));"
                "print(json.dumps({'type':'result','subtype':'success',"
                "'is_error':False,'result':'answer'}))"
            )
            def gateway(*positional, **kwargs):
                return Gateway(*positional, **kwargs, transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json={
                        "type": "message", "id": "msg_test", "model": "model",
                        "content": [{"type": "text", "text": "answer"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 2},
                    }),
                ))
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "host-secret"}), \
                    patch.object(agents.frame_utils, "load_cached_frames", return_value=[frame]), \
                    patch.object(agents, "Gateway", side_effect=gateway), \
                    patch.object(agents, "harness_command", return_value=(
                        ["/opt/harness", "-c", script], "",
                    )) as harness:
                agents.run(args)
                agents.run(args)
                self.assertEqual(harness.call_count, 1)
            result = agents.api.read_csv(args.output)[0]
            self.assertEqual(result["prediction"], "answer")
            self.assertEqual(result["harness"], "claude")
            self.assertEqual(result["requests"], "1")
            self.assertEqual((result["input_tokens"], result["output_tokens"]), ("10", "2"))
            event = json.loads(agents.run_paths.usage_path(args.output).read_text())
            self.assertEqual(event["stage"], "agent")
            self.assertEqual(event["usage"]["input_tokens"], 10)
            self.assertNotIn("host-secret", json.dumps(event))
            self.assertNotIn("private-metadata", json.dumps(event))
            traces = args.output.with_suffix(".traces.jsonl").read_text().splitlines()
            self.assertEqual(len(traces), 1)
            trace = json.loads(traces[0])
            self.assertEqual((trace["video"], trace["status"], trace["exit_code"]),
                             ("clip.mp4", "success", 0))
            self.assertEqual(trace["events"][0], {"type": "user", "image": "<image frame 0>"})
            self.assertEqual(trace["events"][-1]["result"], "answer")
            self.assertNotIn("host-secret", traces[0])
            self.assertNotIn(frame[:40], traces[0])
            meta = json.loads(args.output.with_suffix(".meta.json").read_text())
            self.assertNotIn("trace", json.dumps(meta))


class ReachabilityTests(unittest.TestCase):
    def test_check_model_classifies_key_model_and_network(self):
        ok = httpx.MockTransport(lambda request: httpx.Response(200, json={"id": "m"}))
        self.assertEqual(agents.check_model("anthropic", "m", "k", transport=ok), "m")
        for status in (401, 403, 404):
            bad = httpx.MockTransport(lambda request, status=status: httpx.Response(
                status, json={"error": {"type": "not_found_error"}}))
            with self.assertRaisesRegex(agents.Unreachable, f"HTTP {status}"):
                agents.check_model("openai", "m", "k", transport=bad)

        def down(request):
            raise httpx.ConnectError("no route")
        with self.assertRaisesRegex(agents.Unreachable, "unreachable"):
            agents.check_model("openai", "m", "k", transport=httpx.MockTransport(down))

    def test_fatal_statuses(self):
        self.assertEqual(agents.fatal_upstream(["http_200", "http_429", "http_529", "timeout"]), [])
        self.assertTrue(agents.fatal_upstream(["http_200", "http_404"]))
        self.assertTrue(agents.fatal_upstream(["gateway_error_ConnectError"]))

    def run_rows(self, root, count, upstream, script, extra=()):
        (root / "clip.mp4").touch()
        rows = [{"video": f"clip{i}.mp4", "ground_truth": "A", "question_text": "q"}
                for i in range(count)]
        for row in rows:
            (root / row["video"]).touch()
        input_path = root / "test.csv"
        agents.api.save_csv(input_path, rows, ["video", "ground_truth", "question_text"])
        args = agents.parse_args([
            "--harness", "claude", "--model", "model", "--binary", "/usr/bin/python3",
            "--input", str(input_path), "--video-root", str(root), "--limit", str(count),
            "--rpm", "600", "--output", str(root / "pred.csv"), *extra,
        ])

        def gateway(*positional, **kwargs):
            return Gateway(*positional, **kwargs, transport=httpx.MockTransport(upstream))
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "host-secret"}), \
                patch.object(agents.frame_utils, "load_cached_frames", return_value=["anBlZw=="]), \
                patch.object(agents, "Gateway", side_effect=gateway), \
                patch.object(agents, "harness_command", return_value=(
                    ["/opt/harness", "-c", script], "")) as harness, \
                self.assertRaises(SystemExit) as stop:
            agents.run(args)
        return stop.exception.code, harness.call_count, agents.api.read_csv(args.output)

    def test_provider_rejection_aborts_immediately(self):
        script = (
            "import urllib.request,json,sys\n"
            "req=urllib.request.Request('http://127.0.0.1:8080/v1/messages',"
            "data=json.dumps({'model':'model','max_tokens':5}).encode(),"
            "headers={'content-type':'application/json'})\n"
            "try: urllib.request.urlopen(req)\n"
            "except Exception: sys.exit(1)\n"
        )
        with tempfile.TemporaryDirectory(prefix="morse-agent-abort-") as directory:
            code, calls, rows = self.run_rows(
                Path(directory), 3,
                lambda request: httpx.Response(404, json={"error": {"type": "not_found_error"}}),
                script,
            )
        self.assertEqual(code, agents.EXIT_UNREACHABLE)
        self.assertEqual(calls, 1)
        self.assertEqual(len(rows), 1)
        self.assertIn("http_404", rows[0]["prediction"])

    def test_consecutive_harness_failures_abort(self):
        with tempfile.TemporaryDirectory(prefix="morse-agent-abort-") as directory:
            code, calls, rows = self.run_rows(
                Path(directory), 4, lambda request: httpx.Response(200, json={}),
                "import sys; sys.exit(2)", ("--max-consecutive-failures", "2"),
            )
        self.assertEqual(code, agents.EXIT_UNREACHABLE)
        self.assertEqual(calls, 2)
        self.assertEqual([row["status"] for row in rows], ["failed", "failed"])


if __name__ == "__main__":
    unittest.main()
