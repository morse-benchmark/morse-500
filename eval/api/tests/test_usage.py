from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
import openai
import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimates import usage
from models import run_models_eval as api


class UsageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prices = json.loads(usage.PRICES.read_text())

    def test_concurrent_appends_unknown_usage_and_resume_are_retained(self):
        ledger = usage.Ledger(self.root / "pred.csv", "openai", "gpt-6-astra", "predict", "chat")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: ledger.record(None, 1, "api_error_429", 1), range(20)))
        resumed = usage.Ledger(self.root / "pred.csv", "openai", "gpt-6-astra", "predict", "chat")
        resumed.record(None, 1, "api_error_None", 1)
        events = [json.loads(line) for line in ledger.path.read_text().splitlines()]
        self.assertEqual(len(events), 21)
        self.assertEqual(len({event["run_id"] for event in events}), 2)
        result = usage.summarize([ledger.path, ledger.path], self.prices)[0]
        self.assertEqual(result["requests"], 21)
        self.assertEqual(result["unknown_usage_requests"], 21)
        self.assertIsNone(result["standard_rate_estimate_usd"])

    def test_token_totals_include_anthropic_cache_but_not_double_count_openai(self):
        self.assertEqual(usage.token_totals({
            "provider": "anthropic", "usage": {
                "input_tokens": 10, "output_tokens": 5,
                "cache_read_input_tokens": 20, "cache_creation_input_tokens": 30,
            },
        }), (60, 5))
        self.assertEqual(usage.token_totals({
            "provider": "openai", "usage": {
                "prompt_tokens": 100, "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 20},
                "completion_tokens_details": {"reasoning_tokens": 40},
            },
        }), (100, 50))

    def test_pricing_reproduces_central_estimates_and_unknown_rates_are_null(self):
        expected = {
            "openai/gpt-6-astra": 46.25, "openai/gpt-6.1-sol": 9.25,
            "openai/gpt-6-luna": .4625, "openai/gpt-5.3-codex": 8.75,
            "anthropic/claude-fable-5-1": 46.25, "anthropic/claude-mythos-5-1": 46.25,
            "anthropic/claude-opus-5-5": 18.50, "anthropic/claude-sonnet-5-5": 9.25,
        }
        path = self.root / "test.usage.jsonl"
        for name, cost in expected.items():
            provider, model = name.split("/")
            event = {
                "event_id": name, "provider": provider, "model": model,
                "stage": "predict", "usage": {"input_tokens": 4_000_000, "output_tokens": 125_000},
            }
            path.write_text(json.dumps(event) + "\n")
            result = usage.summarize([path], self.prices)[0]
            self.assertAlmostEqual(result["standard_rate_estimate_usd"], cost)
        event["model"] = "unknown-model"
        path.write_text(json.dumps(event) + "\n")
        self.assertIsNone(usage.summarize([path], self.prices)[0]["standard_rate_estimate_usd"])

    def test_responses_payload_and_truncated_usage_survive_failure(self):
        args = api.parse_args([
            "predict", "--provider", "openai", "--model", "gpt-5.3-codex",
            "--openai-endpoint", "responses", "--reasoning-effort", "low",
            "--output", str(self.root / "pred.csv"),
        ])
        captured = []
        def handler(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "resp_test", "object": "response", "created_at": 0,
                "model": "gpt-5.3-codex", "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [], "parallel_tool_calls": True,
                "tool_choice": "auto", "tools": [],
                "usage": {
                    "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
                    "input_tokens_details": {"cached_tokens": 20},
                    "output_tokens_details": {"reasoning_tokens": 50},
                },
            })
        client = openai.OpenAI(
            api_key="test-only", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only"}), \
                patch.object(api.openai, "OpenAI", return_value=client):
            provider = api.Provider(args)
        try:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                provider.query("question", ["image"], example_id="hashed-example")
        finally:
            provider.close()
        body = captured[0]
        self.assertEqual(body["input"][0]["content"][1]["type"], "input_image")
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertFalse(body["store"])
        events = [json.loads(line) for line in provider.ledger.path.read_text().splitlines()]
        self.assertEqual(events[0]["status"], "incomplete")
        self.assertEqual(events[0]["example_id"], "hashed-example")
        self.assertNotIn("question", provider.ledger.path.read_text())
        result = usage.summarize([provider.ledger.path], self.prices)[0]
        self.assertEqual(result["input_tokens"], 100)
        self.assertEqual(result["output_tokens"], 50)
        self.assertAlmostEqual(result["standard_rate_estimate_usd"], .000875)
        with contextlib.redirect_stdout(io.StringIO()):
            usage.main([str(self.root), "--output", str(self.root / "report.json")])
        self.assertTrue((self.root / "report.json").is_file())

    def test_successful_responses_request_returns_text_and_usage(self):
        args = api.parse_args([
            "predict", "--provider", "openai", "--model", "gpt-5.3-codex",
            "--openai-endpoint", "responses", "--output", str(self.root / "pred.csv"),
        ])
        def handler(request):
            return httpx.Response(200, json={
                "id": "resp_test", "object": "response", "created_at": 0,
                "model": "gpt-5.3-codex", "status": "completed",
                "output": [{
                    "id": "msg_test", "type": "message", "role": "assistant",
                    "status": "completed", "content": [
                        {"type": "output_text", "text": "answer", "annotations": []},
                    ],
                }],
                "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
                "usage": {
                    "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            })
        client = openai.OpenAI(
            api_key="test-only", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only"}), \
                patch.object(api.openai, "OpenAI", return_value=client):
            provider = api.Provider(args)
        try:
            self.assertEqual(provider.query("question", []), ("answer", 10, 2))
        finally:
            provider.close()

    def test_chat_and_anthropic_record_retries_and_truncated_usage(self):
        for name in ("openai", "anthropic"):
            args = api.parse_args([
                "predict", "--provider", name,
                "--model", "gpt-6-luna" if name == "openai" else "claude-opus-5-5",
                "--output", str(self.root / f"{name}.csv"),
            ])
            if name == "openai":
                body = {
                    "id": "chat_test", "object": "chat.completion", "created": 0,
                    "model": args.model, "choices": [{
                        "index": 0, "finish_reason": "length",
                        "message": {"role": "assistant", "content": "partial"},
                    }], "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                }
                factory = openai.OpenAI
                module = api.openai
                class_name = "OpenAI"
            else:
                body = {
                    "id": "msg_test", "type": "message", "role": "assistant",
                    "model": args.model, "content": [{"type": "text", "text": "partial"}],
                    "stop_reason": "max_tokens", "stop_sequence": None,
                    "usage": {
                        "input_tokens": 80, "output_tokens": 50,
                        "cache_read_input_tokens": 20,
                    },
                }
                factory = anthropic.Anthropic
                module = api.anthropic
                class_name = "Anthropic"
            attempts = []
            def handler(request):
                attempts.append(request)
                if len(attempts) == 1:
                    return httpx.Response(
                        429, json={"error": {"type": "rate_limit_error", "message": "limited"}},
                    )
                return httpx.Response(200, json=body)
            client = factory(
                api_key="test-only", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            )
            with patch.dict("os.environ", {
                "OPENAI_API_KEY": "test-only", "ANTHROPIC_API_KEY": "test-only",
            }), patch.object(module, class_name, return_value=client):
                provider = api.Provider(args)
            try:
                with patch.object(provider.limiter, "wait"), patch.object(api.time, "sleep"):
                    with self.assertRaisesRegex(ValueError, "Incomplete/refused"):
                        provider.query("question", [])
            finally:
                provider.close()
            result = usage.summarize([provider.ledger.path], self.prices)[0]
            self.assertEqual(result["requests"], 2)
            self.assertEqual(result["unknown_usage_requests"], 1)
            self.assertEqual(result["input_tokens"], 100)
            self.assertEqual(result["output_tokens"], 50)


if __name__ == "__main__":
    unittest.main()
