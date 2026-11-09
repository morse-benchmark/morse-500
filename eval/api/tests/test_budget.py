import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents import run_agents_eval as agents
from agents.gateway import Gateway
from lib import budget as budget_module
from lib.budget import Budget, BudgetExceeded
from estimates import usage


def write_prices(root):
    path = root / "prices.json"
    path.write_text(json.dumps({"models": {
        "anthropic/model": {"input": 10.0, "output": 50.0},
        "openai/model": {"input": 1.0, "output": 5.0},
    }}))
    return path


def reserve_many(path, prices, count, results):
    budget = Budget(path, 50, "anthropic", "model", prices)
    results.put(sum(budget.reserve(1.0) is not None for _ in range(count)))


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.prices = write_prices(self.root)
        self.path = self.root / "budget.jsonl"

    def tearDown(self):
        self.directory.cleanup()

    def budget(self, limit=10.0, provider="anthropic"):
        return Budget(self.path, limit, provider, "model", self.prices)

    def test_reserve_settle_and_refuse(self):
        budget = self.budget(limit=1.0)
        key = budget.reserve(0.6)
        self.assertIsNotNone(key)
        self.assertIsNone(budget.reserve(0.5))  # 0.6 open + 0.5 > 1.0
        usage = {"input_tokens": 10_000, "output_tokens": 1_000,
                 "cache_read_input_tokens": 10_000}
        self.assertAlmostEqual(budget.settle(key, 0.6, usage, "end_turn", True), 0.25)
        self.assertEqual(budget.spent(), (0.25, 0))
        self.assertIsNotNone(budget.reserve(0.75))
        self.assertIsNone(budget.reserve(0.01))
        with self.assertRaises(BudgetExceeded):
            budget.require(0.01)
        # Another job sharing the file sees the same commitments.
        self.assertAlmostEqual(sum(self.budget(limit=1.0).spent()), 1.0)

    def test_settlement_is_conservative_when_usage_may_be_partial(self):
        budget = self.budget()
        partial = {"input_tokens": 100, "output_tokens": 1}
        cases = [
            (partial, "streaming", True, 2.0),            # killed mid-stream
            (partial, "timeout", True, 2.0),
            (None, "http_200", True, 2.0),                # accepted, usage never seen
            (partial, "gateway_error_ReadError", True, 2.0),
            (None, "http_429", False, 0.0),               # provider refused: not billed
            (None, "gateway_error_ConnectError", False, 0.0),
        ]
        for usage, status, accepted, expected in cases:
            key = budget.reserve(2.0)
            self.assertAlmostEqual(budget.settle(key, 2.0, usage, status, accepted), expected,
                                   msg=status)

    def test_unsettled_reservations_stay_counted_and_torn_tail_ignored(self):
        budget = self.budget()
        budget.reserve(3.0)
        with self.path.open("a") as stream:
            stream.write('{"type": "settle", "id": "x"')  # killed writer
        self.assertEqual(budget.spent(), (0, 3.0))

    def test_cost_matches_usage_report(self):
        budget = self.budget(provider="openai")
        event = {"provider": "openai", "model": "model", "stage": "agent", "event_id": "1",
                 "status": "completed",
                 "usage": {"input_tokens": 123_456, "output_tokens": 7_890}}
        ledger = self.root / "x.usage.jsonl"
        ledger.write_text(json.dumps(event) + "\n")
        prices = json.loads(self.prices.read_text())
        report = usage.summarize([ledger], prices)[0]["standard_rate_estimate_usd"]
        self.assertAlmostEqual(budget.usage_cost(event["usage"]), report)

    def test_worst_case_covers_images_text_and_full_output(self):
        budget = self.budget()
        image = "/9j/" + "A" * 20_000
        body = {"messages": [{"content": [{"type": "text", "text": "x" * 3000},
                                          {"type": "image", "source": {"data": image}}]}]}
        cost = budget.worst_case(body, 8192)
        self.assertGreater(cost, budget.cost(1000 + budget_module.TOKENS_PER_IMAGE, 8192) - 1e-9)
        self.assertLess(cost, budget.cost(2000 + budget_module.TOKENS_PER_IMAGE, 8192))

    def test_concurrent_processes_never_overcommit(self):
        results = multiprocessing.Queue()
        workers = [multiprocessing.Process(
            target=reserve_many, args=(self.path, self.prices, 20, results)) for _ in range(6)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(30)
        self.assertEqual(sum(results.get(timeout=5) for _ in workers), 50)
        self.assertEqual(self.budget(limit=50).spent(), (0, 50.0))

    def test_concurrent_threads_never_overcommit(self):
        budgets = [self.budget(limit=10.0) for _ in range(8)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            reservations = list(pool.map(
                lambda index: budgets[index % len(budgets)].reserve(1.0), range(40),
            ))
        self.assertEqual(sum(key is not None for key in reservations), 10)
        self.assertEqual(self.budget(limit=10.0).spent(), (0, 10.0))


class GatewayBudgetTests(unittest.TestCase):
    def test_gateway_reserves_settles_and_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = Budget(root / "budget.jsonl", 0.42, "anthropic", "model", write_prices(root))
            calls = []

            def respond(request):
                calls.append(json.loads(request.content))
                return httpx.Response(200, json={
                    "type": "message", "id": "m", "model": "model", "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1000, "output_tokens": 100},
                })
            ledger = usage.Ledger(root / "pred.csv", "anthropic", "model", "agent", "claude")
            gateway = Gateway(root / "api.sock", "anthropic", "model", "key", ledger,
                              agents.api.RateLimiter(6000), 30, 60, 8192, "ex",
                              transport=httpx.MockTransport(respond), budget=budget)
            body = json.dumps({"model": "model", "max_tokens": 8192,
                               "messages": [{"role": "user", "content": "hi"}]})
            with gateway.running():
                client = httpx.Client(transport=httpx.HTTPTransport(uds=str(root / "api.sock")))
                first = client.post("http://local/v1/messages", content=body,
                                    headers={"content-type": "application/json"})
                self.assertEqual(first.status_code, 200)
                # Worst case ~$0.41 no longer fits in the remaining $0.405.
                second = client.post("http://local/v1/messages", content=body,
                                     headers={"content-type": "application/json"})
                client.close()
            self.assertEqual(second.status_code, 400)
            self.assertIn("budget reached", second.text)
            self.assertEqual(len(calls), 1)
            self.assertEqual((gateway.usd_denied, gateway.errors), (1, 0))
            self.assertAlmostEqual(budget.spent()[0], 0.015)
            self.assertEqual(budget.spent()[1], 0)


class RunnerBudgetTests(unittest.TestCase):
    def test_runner_stops_with_checkpoint_and_resumes_with_higher_budget(self):
        with tempfile.TemporaryDirectory(prefix="morse-budget-") as directory:
            root = Path(directory)
            prices = write_prices(root)
            rows = []
            for name in ("a.mp4", "b.mp4", "c.mp4"):
                (root / name).touch()
                rows.append({"video": name, "ground_truth": "x", "question_text": "q"})
            agents.api.save_csv(root / "test.csv", rows, ["video", "ground_truth", "question_text"])
            script = (
                "import urllib.request,json,urllib.error\n"
                "for _ in range(2):\n"
                " req=urllib.request.Request('http://127.0.0.1:8080/v1/messages',"
                "data=json.dumps({'model':'model','max_tokens':1000}).encode(),"
                "headers={'content-type':'application/json'})\n"
                " try: urllib.request.urlopen(req).read()\n"
                " except urllib.error.HTTPError as e: print(json.dumps({'type':'denied','code':e.code})); raise SystemExit(1)\n"
                "print(json.dumps({'type':'result','subtype':'success','is_error':False,'result':'A'}))"
            )

            def gateway(*positional, **kwargs):
                return Gateway(*positional, **kwargs, transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json={
                        "type": "message", "id": "m", "model": "model", "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "A"}],
                        "usage": {"input_tokens": 1000, "output_tokens": 1000},
                    })))

            def run(limit):
                args = agents.parse_args([
                    "--harness", "claude", "--model", "model", "--binary", "/usr/bin/python3",
                    "--input", str(root / "test.csv"), "--video-root", str(root),
                    "--limit", "3", "--rpm", "6000", "--max-output-tokens", "1000",
                    "--budget-usd", str(limit), "--output", str(root / "out" / "pred.csv"),
                ])
                with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}), \
                        patch.object(budget_module, "PRICES", prices), \
                        patch.object(agents, "Budget", lambda *a: Budget(*a, prices)), \
                        patch.object(agents.frame_utils, "load_cached_frames", return_value=["anBlZw=="]), \
                        patch.object(agents, "Gateway", side_effect=gateway), \
                        patch.object(agents, "harness_command",
                                     return_value=(["/opt/harness", "-c", script], "")):
                    agents.run(args)
                return args.output

            # Each request: reserve <= $0.06 worst case, settle $0.06 actual. Two per example.
            with self.assertRaises(BudgetExceeded):
                run(0.2)
            output = root / "out" / "pred.csv"
            results = agents.api.read_csv(output)
            self.assertEqual([r["status"] for r in results], ["success", "failed"])
            self.assertIn("USD budget", results[1]["prediction"])
            traces = [json.loads(line) for line in
                      output.with_suffix(".traces.jsonl").read_text().splitlines()]
            self.assertEqual(traces[-1]["events"][-1], {"type": "denied", "code": 400})
            budget = Budget(root / "out/run_usage/budget.jsonl", 0.2, "anthropic", "model", prices)
            settled, open_usd = budget.spent()
            self.assertLessEqual(settled + open_usd, 0.2)
            self.assertEqual(open_usd, 0)

            run(1.0)  # Approved increase: resumes, retries the interrupted example.
            results = agents.api.read_csv(output)
            self.assertEqual([r["status"] for r in results], ["success"] * 3)
            self.assertAlmostEqual(sum(budget.spent()), 0.06 * 7)


if __name__ == "__main__":
    unittest.main()
