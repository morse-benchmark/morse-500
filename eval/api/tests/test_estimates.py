import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimates import estimate_agent_costs as est
from estimates import make_subset
from estimates import read_estimate


def rows(spec):
    out, number = [], 0
    for category, count in spec.items():
        for _ in range(count):
            number += 1
            out.append(dict(id=str(number), video=f"v{number}.mp4", query="q",
                            question_text="t", ground_truth="A", category=category))
    return out


def write_csv(path, data, fields=None):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(data[0]))
        writer.writeheader()
        writer.writerows(data)


class SubsetTests(unittest.TestCase):
    def test_equal_allocation_and_determinism(self):
        data = rows({"a": 108, "b": 100, "c": 84, "d": 80, "e": 64, "f": 64})
        subset, counts = make_subset.stratified_subset(data, 24, 0)
        self.assertEqual(set(counts.values()), {4})
        self.assertEqual(Counter(r["category"] for r in subset), Counter({k: 4 for k in "abcdef"}))
        self.assertEqual(subset, make_subset.stratified_subset(data, 24, 0)[0])
        self.assertNotEqual(subset, make_subset.stratified_subset(data, 24, 1)[0])
        ids = [int(r["id"]) for r in subset]
        self.assertEqual(ids, sorted(ids))

    def test_remainder_goes_to_largest(self):
        counts = make_subset.allocate({"a": 10, "b": 50, "c": 30}, 5)
        self.assertEqual(counts, {"b": 2, "c": 2, "a": 1})
        with self.assertRaises(ValueError):
            make_subset.allocate({"a": 10, "b": 5}, 1)

    def test_cli_refuses_different_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root / "full.csv", rows({"a": 20, "b": 20}))
            args = ["--input", str(root / "full.csv"), "--output", str(root / "s.csv"), "--size", "4"]
            make_subset.main(args)
            make_subset.main(args)  # identical: allowed (resume)
            with self.assertRaises(SystemExit):
                make_subset.main(args[:-1] + ["6"])


class EstimateTests(unittest.TestCase):
    def test_read_estimate_sums_selected_point_or_upper_bounds(self):
        report = {"models": [
            {"provider": "openai", "model": "a", "usd": {"estimate": 10, "interval": [8, 12]}},
            {"provider": "openai", "model": "b", "usd": {"estimate": 20, "interval": [18, 22]}},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "estimate.json"
            path.write_text(json.dumps(report))
            actual = json.loads(path.read_text())
            self.assertEqual(read_estimate.total_cost(actual, "openai", ["a", "b"], "point"), 30)
            self.assertEqual(read_estimate.total_cost(actual, "openai", ["a", "b"], "upper"), 34)
            with self.assertRaises(SystemExit):
                read_estimate.main([str(path), "anthropic", "a"])

    def test_stratified_projection_and_flags(self):
        full = rows({"a": 300, "b": 200})
        subset = [full[0], full[1], full[300], full[301]]
        prices = {"as_of": "x", "models": {"anthropic/m": {"input": 1.0, "output": 10.0}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root / "full.csv", full)
            write_csv(root / "subset.csv", subset)
            pred = root / "pred_sz512_claude_m_fps4_max32.csv"
            out_rows = []
            for row, seconds, text in zip(subset, (60, 120, 600, 600), (
                "A", "B", "C", "ERROR: Harness exhausted request budget (per-example request/time cap)",
            )):
                out_rows.append(dict(video=row["video"], prediction=text, elapsed_seconds=seconds))
            write_csv(pred, out_rows)
            pred.with_suffix(".meta.json").write_text(json.dumps(
                dict(harness="claude", model="m", max_requests=30, seconds=1800)))
            # Per-example USD: a: 1.0, 3.0 (mean 2); b: 10.0 (mean 10, one capped example free).
            events = [
                (subset[0], 1_000_000, 0), (subset[1], 1_000_000, 200_000),
                (subset[2], 0, 1_000_000), (subset[3], None, None),
                (dict(video="other.mp4"), 1, 1),
            ]
            ledger_path = pred.parent / "run_usage" / pred.with_suffix(".usage.jsonl").name
            ledger_path.parent.mkdir(exist_ok=True)
            with ledger_path.open("w") as stream:
                for number, (row, tokens_in, tokens_out) in enumerate(events):
                    usage = None if tokens_in is None else dict(input_tokens=tokens_in, output_tokens=tokens_out)
                    stream.write(json.dumps(dict(
                        event_id=str(number), provider="anthropic", model="m",
                        example_id=est.example_hash(row["video"]), usage=usage, status="ok",
                    )) + "\n")
                stream.write('{"partial')  # mid-write line is tolerated
            [report] = est.estimate(root, root / "subset.csv", root / "full.csv",
                                    prices, 500, 200, 0, 0.9)
        # 300 * mean(1, 3) + 200 * mean(10, 0) = 600 + 1000
        self.assertAlmostEqual(report["usd"]["estimate"], 1600)
        self.assertAlmostEqual(report["hours"]["estimate"], (300 * 90 + 200 * 600) / 3600)
        low, high = report["usd"]["interval"]
        self.assertLessEqual(low, 1600)
        self.assertGreaterEqual(high, 1600)
        self.assertEqual(report["capped_examples"], 1)
        self.assertEqual(report["unknown_usage_requests"], 1)
        self.assertEqual(report["unattributed_ledger_events"], 1)
        self.assertIn("| a | 2 | 300 |", est.render([report], 500, 0.9))

    def test_direct_model_metadata_is_included_in_cost_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = rows({"a": 2})
            write_csv(root / "full.csv", full)
            write_csv(root / "subset.csv", full)
            prediction = root / "pred_sz512_openai_model_fps4_max32.csv"
            write_csv(prediction, [
                {"video": row["video"], "prediction": "answer", "elapsed_seconds": "5"}
                for row in full
            ])
            prediction.with_suffix(".meta.json").write_text(json.dumps({
                "provider": "openai", "model": "model",
            }))
            ledger_path = prediction.parent / "run_usage" / prediction.with_suffix(".usage.jsonl").name
            ledger_path.parent.mkdir(exist_ok=True)
            with ledger_path.open("w") as stream:
                for row in full:
                    stream.write(json.dumps({
                        "event_id": row["video"], "provider": "openai", "model": "model",
                        "example_id": est.example_hash(row["video"]),
                        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                    }) + "\n")
            prices = {"as_of": "x", "models": {
                "openai/model": {"input": 1.0, "output": 5.0},
            }}
            [report] = est.estimate(root, root / "subset.csv", root / "full.csv",
                                    prices, 2, 20, 0, 0.9)
        self.assertEqual(report["mode"], "model")
        self.assertEqual(report["provider"], "openai")
        self.assertEqual(report["harness"], "openai")
        self.assertAlmostEqual(report["usd"]["estimate"], 0.0003)


if __name__ == "__main__":
    unittest.main()
