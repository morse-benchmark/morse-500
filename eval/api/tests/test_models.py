import base64
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import budget as budget_module, common, frames
from lib.budget import BudgetExceeded
from models import run_models_eval as api
from utils import check_api, prepare_failed_rows as recovery
from tests.session_helper import session_env


class SamplingTests(unittest.TestCase):
    def test_quarter_second_grid_and_uniform_cap(self):
        np.testing.assert_array_equal(frames.frame_times(1, 4, 32), [0, .25, .5, .75])
        times = frames.frame_times(20, 4, 32)
        self.assertEqual(len(times), 32)
        self.assertEqual(times[0], 0)
        self.assertEqual(times[-1], 19.75)
        self.assertTrue(np.all(times * 4 == np.floor(times * 4)))
        self.assertTrue(np.all(np.diff(times) > 0))

    def test_resize_preserves_aspect_ratio_without_upscaling(self):
        for width, height, expected in [
            (1920, 1080, (512, 288)),
            (1080, 1920, (288, 512)),
            (128, 64, (128, 64)),
        ]:
            encoded = frames.encode_frame(np.zeros((height, width, 3), dtype=np.uint8), 512)
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                self.assertEqual(image.size, expected)
                self.assertEqual(image.format, "JPEG")

    def test_invalid_duration(self):
        for duration in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                frames.frame_times(duration, 4, 32)

    def test_provider_payloads(self):
        openai = api.content_blocks("openai", "question", ["encoded"])
        self.assertEqual(openai[1]["image_url"]["url"], "data:image/jpeg;base64,encoded")
        self.assertEqual(openai[1]["image_url"]["detail"], "high")
        anthropic = api.content_blocks("anthropic", "question", ["encoded"])
        self.assertEqual(anthropic[1]["source"]["data"], "encoded")
        self.assertEqual(anthropic[1]["source"]["media_type"], "image/jpeg")

    def test_rate_limiter_spaces_requests(self):
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
        with patch.object(common.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(common.time, "sleep", side_effect=sleep):
            limiter = common.RateLimiter(20)
            limiter.wait()
            limiter.wait()
            limiter.wait()
        self.assertEqual(clock[0], 6)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "test.csv"
        self.output = self.root / "pred.csv"
        self.rows = [
            {"video": f"{i}.mp4", "ground_truth": "None", "question_text": "question"}
            for i in range(2)
        ]
        common.save_csv(self.input, self.rows, list(self.rows[0]))
        for row in self.rows:
            (self.root / row["video"]).touch()
        self.args = api.parse_args([
            "predict", "--provider", "openai", "--model", "test-model",
            "--input", str(self.input), "--video-root", str(self.root),
            "--output", str(self.output), "--concurrency", "1",
        ])

    def test_select_failed_rows_in_source_order_and_keep_original_indices(self):
        source_rows = [
            {"id": str(index), "video": f"{index}.mp4", "ground_truth": "A",
             "question_text": "question"}
            for index in range(1, 5)
        ]
        predictions = [
            {"video": "1.mp4", "prediction": "answer"},
            {"video": "2.mp4", "prediction": "ERROR: truncated"},
            {"video": "4.mp4", "prediction": ""},
        ]
        failed = recovery.select_failed_rows(source_rows, predictions)
        self.assertEqual([row["video"] for row in failed], ["2.mp4", "3.mp4", "4.mp4"])
        self.assertEqual([row["idx"] for row in failed], ["2", "3", "4"])

    def test_select_failed_rows_rejects_unexpected_or_duplicate_predictions(self):
        source_rows = [dict(self.rows[0])]
        for predictions in (
            [{"video": "unexpected.mp4", "prediction": "answer"}],
            [{"video": "0.mp4", "prediction": "answer"},
             {"video": "0.mp4", "prediction": "answer"}],
        ):
            with self.assertRaises(ValueError):
                recovery.select_failed_rows(source_rows, predictions)

    def test_prediction_output_keeps_recovery_benchmark_index(self):
        source = self.root / "failed.csv"
        common.save_csv(source, [dict(self.rows[0], idx="42")],
                     list(self.rows[0]) + ["idx"])
        self.args.input = source
        self.args.output = self.root / "recovery.csv"
        provider = Mock()
        provider.query.return_value = ("fixed", 10, 2)
        with patch.object(api, "Provider", return_value=provider), \
                patch.object(frames, "sample_video", return_value=["image"]):
            api.run(self.args)
        self.assertEqual(common.read_csv(self.args.output)[0]["idx"], "42")

    def test_failed_row_batch_uses_separate_output_and_higher_budget(self):
        fake_python = self.root / "python"
        fake_python.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['CALL_LOG'], 'a') as f:\n"
            "    f.write(json.dumps(args) + '\\n')\n"
            "if args[0].endswith('lib/run_paths.py'):\n"
            "    os.execv(sys.executable, [sys.executable, *args])\n"
            "if args[0].endswith('prepare_failed_rows.py'):\n"
            "    out = args[args.index('--output') + 1]\n"
            "    Path(out).parent.mkdir(parents=True, exist_ok=True)\n"
            "    open(out, 'w').write('video,ground_truth,question_text,idx\\\\n0.mp4,A,q,1\\\\n')\n"
        )
        fake_python.chmod(0o700)
        call_log = self.root / "calls.jsonl"
        env = session_env(
            self.root, PYTHON=str(fake_python), CALL_LOG=str(call_log),
            REPO=str(self.root), OPENAI_API_KEY="fake-test-key",
            ORIGINAL_RESULTS=str(self.root / "eval/results/original"), RUN_NAME="recovery",
        )
        subprocess.run(
            ["bash", str(Path(api.__file__).parents[1] / "scripts" / "extras" / "retry_failed.sh"),
             "openai", "test-model"],
            cwd=self.root, env=env, check=True, capture_output=True, text=True,
        )
        calls = [json.loads(line) for line in call_log.read_text().splitlines()]
        calls = [call for call in calls if not call[0].endswith("lib/run_paths.py")]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], "eval/api/utils/prepare_failed_rows.py")
        predict = calls[1]
        self.assertEqual(predict[0], "eval/api/models/run_models_eval.py")
        for call in calls:
            self.assertTrue((common.REPO / call[0]).is_file())
        self.assertEqual(predict[predict.index("--max-output-tokens") + 1], "8192")
        self.assertEqual(predict[predict.index("--input") + 1],
                         calls[0][calls[0].index("--output") + 1])
        output = Path(predict[predict.index("--output") + 1])
        self.assertIn("/eval/results/recovery/pred_retry_", str(output))
        self.assertNotEqual(output, Path(calls[0][calls[0].index("--predictions") + 1]))

    def test_checkpoint_resume_configuration_guard_and_literal_none(self):
        provider = Mock()
        provider.query.side_effect = [("None", 12, 3), ValueError("truncated")]
        with patch.object(api, "Provider", return_value=provider), \
                patch.object(frames, "sample_video", return_value=["image"]):
            with self.assertRaisesRegex(RuntimeError, "1 failed"):
                api.run(self.args)
        checkpoint = common.read_csv(self.output)
        self.assertEqual(checkpoint[0]["prediction"], "None")
        self.assertTrue(checkpoint[1]["prediction"].startswith("ERROR:"))
        provider.query.side_effect = None
        provider.query.return_value = ("fixed", 20, 4)
        provider.query.reset_mock()
        with patch.object(api, "Provider", return_value=provider), \
                patch.object(frames, "sample_video", return_value=["image"]):
            api.run(self.args)
        self.assertEqual(provider.query.call_count, 1)
        self.assertEqual(len(common.read_csv(self.output)), 2)
        self.args.max_frames = 16
        with self.assertRaisesRegex(ValueError, "configuration/input changed"):
            api.run(self.args)

    def test_resume_accepts_checkpoint_metadata_from_before_reorganization(self):
        source_records = [
            frames.source_identity(self.root / row["video"]) for row in self.rows
        ]
        metadata = {
            "command": "predict", "provider": "openai", "model": "test-model",
            "input": str(self.input), "video_root": str(self.root), "frame_cache": None,
            "fps": 4, "max_frames": 32, "max_side": 512, "max_output_tokens": 4096,
            "reasoning_effort": None, "limit": None, "expected_count": None,
            "input_sha256": hashlib.sha256(self.input.read_bytes()).hexdigest(),
            "protocol_version": 1,
            "video_sources_sha256": hashlib.sha256(
                json.dumps(source_records, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }
        self.output.with_suffix(".meta.json").write_text(json.dumps(metadata))
        existing = {field: "" for field in common.FIELDS}
        existing.update(self.rows[0], idx="1", prediction="previous answer")
        common.save_csv(self.output, [existing], common.FIELDS)

        provider = Mock()
        provider.query.return_value = ("new answer", 10, 2)
        with patch.object(api, "Provider", return_value=provider), \
                patch.object(frames, "sample_video", return_value=["image"]):
            api.run(self.args)
        self.assertEqual(provider.query.call_count, 1)
        self.assertEqual(
            [row["prediction"] for row in common.read_csv(self.output)],
            ["previous answer", "new answer"],
        )

    def test_extract_refuses_partial_or_error_predictions(self):
        common.save_csv(
            self.output, [dict(self.rows[0], prediction="ERROR: failure")],
            list(self.rows[0]) + ["prediction"],
        )
        args = api.parse_args([
            "extract", "--provider", "anthropic", "--model", "judge",
            "--input", str(self.output), "--output", str(self.root / "extract.csv"),
            "--expected-count", "500",
        ])
        with self.assertRaisesRegex(ValueError, "Expected 500"):
            api.run(args)
        args.expected_count = None
        with self.assertRaisesRegex(ValueError, "Predictions contain errors"):
            api.run(args)

    def prepare_args(self):
        return api.parse_args([
            "prepare", "--input", str(self.input), "--video-root", str(self.root),
            "--frame-cache", str(self.root / "cache"),
        ])

    def test_prepare_reuses_identical_frames_without_credentials(self):
        args = self.prepare_args()
        images = [frames.encode_frame(np.zeros((16, 32, 3), dtype=np.uint8), 512)]
        with patch.object(frames, "sample_video", return_value=images) as sample, \
                patch.object(api, "Provider") as provider:
            api.run(args)
            self.assertEqual(sample.call_count, 2)
            api.run(args)
            self.assertEqual(sample.call_count, 2)
            provider.assert_not_called()
        self.assertEqual(frames.load_cached_frames(args, "0.mp4"), images)

    def test_cache_rejects_missing_stale_incompatible_and_corrupt_entries(self):
        args = self.prepare_args()
        with self.assertRaisesRegex(FileNotFoundError, "run prepare first"):
            frames.load_cached_frames(args, "0.mp4")
        with patch.object(frames, "sample_video", return_value=["image"]):
            api.run(args)
        args.max_frames = 16
        with self.assertRaisesRegex(ValueError, "Stale/incompatible"):
            frames.load_cached_frames(args, "0.mp4")
        args.max_frames = 32
        (self.root / "0.mp4").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Stale/incompatible"):
            frames.load_cached_frames(args, "0.mp4")
        path = frames.cache_path(args, "1.mp4")
        record = json.loads(path.read_text())
        record["frames"][0] = "corrupted"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            frames.load_cached_frames(args, "1.mp4")

    def test_two_models_use_prepared_frames_without_decoding(self):
        args = self.prepare_args()
        images = ["image-one", "image-two"]
        with patch.object(frames, "sample_video", return_value=images):
            api.run(args)
        self.args.frame_cache = args.frame_cache
        provider = Mock()
        provider.query.return_value = ("answer", 10, 2)
        with patch.object(frames, "sample_video") as sample, \
                patch.object(api, "Provider", return_value=provider):
            for model in ("model-one", "model-two"):
                self.args.model = model
                self.args.output = self.root / f"{model}.csv"
                api.run(self.args)
                result = common.read_csv(self.args.output)
                self.assertEqual([r["frame_count"] for r in result], ["2", "2"])
            sample.assert_not_called()
        self.assertEqual(provider.query.call_count, 4)
        for call in provider.query.call_args_list:
            self.assertEqual(call.args[1], images)

    def test_missing_cache_fails_before_provider_initialization(self):
        self.args.frame_cache = self.root / "missing"
        with patch.object(api, "Provider") as provider:
            with self.assertRaises(FileNotFoundError):
                api.run(self.args)
            provider.assert_not_called()

    def test_cli_prepare_needs_no_provider_and_requires_cache(self):
        self.assertEqual(self.prepare_args().command, "prepare")
        for argv in [
            ["prepare"],
            ["predict", "--output", "out.csv"],
            ["extract", "--provider", "openai", "--model", "judge",
             "--output", "out.csv", "--frame-cache", "cache"],
        ]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                api.parse_args(argv)

    def test_unified_runner_dispatches_model_mode_without_paid_requests(self):
        fake_python = self.root / "python"
        fake_python.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['CALL_LOG'], 'a') as f:\n"
            " f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if sys.argv[1].endswith('lib/run_paths.py'):\n"
            " os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        )
        fake_python.chmod(0o700)
        log = self.root / "calls.jsonl"
        env = session_env(self.root, PYTHON=str(fake_python), CALL_LOG=str(log),
                   OPENAI_API_KEY="fake-test-key", REPO=str(self.root),
                   DATA_CSV=str(self.args.input), VIDEO_ROOT=str(self.root),
                   OUTPUT_DIR=str(self.root / "eval/results/test"),
                   FRAME_CACHE=str(self.root / "frames"), LIMIT="1")
        subprocess.run(
            ["bash", str(common.REPO / "eval/api/scripts/run_local.sh"),
             "model", "openai", "test-one", "test-two"],
            env=env, check=True, capture_output=True, text=True,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        checks = [call for call in calls if call[0].endswith("utils/check_api.py")]
        self.assertEqual(len(checks), 2)
        self.assertEqual([call[call.index("--model") + 1] for call in checks],
                         ["test-one", "test-two"])
        self.assertTrue(any(call[:2] == ["-c", "import anthropic, moviepy, numpy, openai, pandas, PIL"]
                            for call in calls))
        prepare = next(call for call in calls if call[0].endswith("lib/frames.py"))
        predicts = [call for call in calls if call[0].endswith("models/run_models_eval.py")]
        self.assertEqual(prepare[1], "prepare")
        self.assertEqual(len(predicts), 2)
        self.assertEqual([call[call.index("--model") + 1] for call in predicts],
                         ["test-one", "test-two"])
        self.assertTrue(all(call[call.index("--limit") + 1] == "1" for call in predicts))

    def test_unified_runner_stops_if_frame_preparation_fails(self):
        fake_python = self.root / "python"
        fake_python.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['CALL_LOG'], 'a') as f:\n"
            " f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if sys.argv[1].endswith('lib/run_paths.py'):\n"
            " os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
            "if sys.argv[1].endswith('lib/frames.py'): raise SystemExit(12)\n"
        )
        fake_python.chmod(0o700)
        log = self.root / "calls.jsonl"
        env = session_env(self.root, PYTHON=str(fake_python), CALL_LOG=str(log),
                   OPENAI_API_KEY="fake-test-key", REPO=str(self.root),
                   DATA_CSV=str(self.args.input), VIDEO_ROOT=str(self.root),
                   OUTPUT_DIR=str(self.root / "eval/results/test"),
                   FRAME_CACHE=str(self.root / "frames"), LIMIT="1")
        result = subprocess.run(
            ["bash", str(common.REPO / "eval/api/scripts/run_local.sh"),
             "model", "openai", "test-model"],
            env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 12, result.stderr)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertFalse(any(call[0].endswith("models/run_models_eval.py") for call in calls))

    def test_extraction_uses_no_ground_truth_and_preserves_raw_prediction(self):
        row = dict(self.rows[0], prediction="The answer is None.")
        row["ground_truth"] = "SECRET_GROUND_TRUTH"
        self.assertNotIn("SECRET_GROUND_TRUTH", api.extraction_prompt(row))
        common.save_csv(self.output, [row], list(row))
        args = api.parse_args([
            "extract", "--provider", "anthropic", "--model", "judge",
            "--input", str(self.output), "--output", str(self.root / "extract.csv"),
        ])
        provider = Mock()
        provider.query.return_value = ("None", 100, 1)
        with patch.object(api, "Provider", return_value=provider):
            api.run(args)
        extracted = common.read_csv(args.output)[0]
        self.assertEqual(extracted["extracted_answer"], "None")
        self.assertEqual(extracted["prediction"], row["prediction"])

    def test_cli_rejects_invalid_settings(self):
        for flag, value in [("--fps", "0"), ("--rpm", "nan"), ("--concurrency", "-1")]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                api.parse_args([
                    "predict", "--provider", "openai", "--model", "test",
                    "--output", "unused.csv", flag, value,
                ])

    def test_http_sdk_payloads_and_usage_without_network(self):
        for provider in ("openai", "anthropic"):
            self.args.provider = provider
            self.args.rpm = 100000
            if provider == "openai":
                response = {
                    "id": "test", "object": "chat.completion", "created": 0,
                    "model": "test-model", "choices": [{
                        "index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "answer"},
                    }], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                }
                client_class = api.openai.OpenAI
            else:
                response = {
                    "id": "test", "type": "message", "role": "assistant",
                    "model": "test-model", "content": [{"type": "text", "text": "answer"}],
                    "stop_reason": "end_turn", "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
                client_class = api.anthropic.Anthropic
            captured = []
            def handler(request):
                captured.append(json.loads(request.content))
                return httpx.Response(200, json=response)
            client = client_class(
                api_key="test-not-a-real-key", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            )
            instance = object.__new__(api.Provider)
            instance.ledger = Mock()
            instance.args = self.args
            instance.client = client
            instance.limiter = common.RateLimiter(100000)
            try:
                self.assertEqual(instance.query("question", ["base64"]), ("answer", 10, 2))
            finally:
                instance.close()
            self.assertEqual(captured[0]["model"], "test-model")
            self.assertEqual(len(captured[0]["messages"][0]["content"]), 2)

    def test_model_budget_reserves_before_request_and_settles_metered_usage(self):
        prices = self.root / "prices.json"
        prices.write_text(json.dumps({"models": {
            "openai/test-model": {"input": 1.0, "output": 5.0},
        }}))
        ledger_path = self.root / "budget.jsonl"
        args = SimpleNamespace(
            provider="openai", model="test-model", rpm=6000, attempts=1,
            max_output_tokens=100, reasoning_effort=None, openai_endpoint="chat",
        )
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=2,
                model_dump=lambda mode: {"prompt_tokens": 10, "completion_tokens": 2},
            ),
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="answer"),
            )],
        )
        instance = object.__new__(api.Provider)
        instance.args = args
        instance.limiter = common.RateLimiter(6000)
        instance.ledger = Mock()
        instance.client = Mock()
        instance.client.chat.completions.create.return_value = response
        instance.budget = budget_module.Budget(
            ledger_path, 1.0, "openai", "test-model", prices,
        )
        self.assertEqual(instance.query("question", [], "example"), ("answer", 10, 2))
        self.assertAlmostEqual(instance.budget.spent()[0], 0.00002)

        instance.budget = budget_module.Budget(
            ledger_path.with_name("small-budget.jsonl"), 0.0001,
            "openai", "test-model", prices,
        )
        instance.client.chat.completions.create.reset_mock()
        with self.assertRaises(BudgetExceeded):
            instance.query("question", [], "example")
        instance.client.chat.completions.create.assert_not_called()

    def test_model_budget_cli_defaults_ledger_next_to_output(self):
        args = api.parse_args([
            "predict", "--provider", "openai", "--model", "test-model",
            "--output", str(self.root / "pred.csv"), "--budget-usd", "5",
        ])
        self.assertEqual(args.budget_file, self.root / "run_usage/budget.jsonl")

    def test_scoring_matches_exact_answers_and_merges_causal(self):
        input_dir = self.root / "extract"
        input_dir.mkdir()
        rows = [
            {"video": "physical_reasoning_a_b.mp4", "ground_truth": "None", "extracted_answer": "None"},
            {"video": "causal_reasoning_a_b.mp4", "ground_truth": "A", "extracted_answer": "a"},
            {"video": "temporal_reasoning_a_b.mp4", "ground_truth": "B", "extracted_answer": "B"},
        ]
        common.save_csv(input_dir / "extract_sz512_test.csv", rows, list(rows[0]))
        output = self.root / "scores.csv"
        subprocess.run([
            sys.executable, str(common.REPO / "eval" / "qwen_eval" / "plot_table.py"),
            "--input-dir", str(input_dir), "--output", str(output),
        ], check=True, capture_output=True, text=True)
        result = common.read_csv(output)[0]
        self.assertAlmostEqual(float(result["avg_acc"]), 200 / 3)
        self.assertEqual(float(result["physical_reasoning"]), 100)
        self.assertEqual(float(result["temporal_reasoning"]), 50)

    def test_retryable_errors_are_rate_limited_again(self):
        self.args.attempts = 2
        instance = object.__new__(api.Provider)
        instance.ledger = Mock()
        instance.args = self.args
        instance.limiter = Mock()
        instance.client = Mock()
        response = Mock()
        response.choices = [
            Mock(finish_reason="stop", message=Mock(content="answer"))
        ]
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 2
        request = httpx.Request("POST", "https://example.invalid")
        error = api.openai.RateLimitError(
            "limited",
            response=httpx.Response(429, headers={"retry-after": "7"}, request=request),
            body=None,
        )
        instance.client.chat.completions.create.side_effect = [error, response]
        with patch.object(api.time, "sleep") as sleep:
            self.assertEqual(instance.query("question", []), ("answer", 10, 2))
        sleep.assert_called_once_with(7)
        self.assertEqual(instance.limiter.wait.call_count, 2)

    def test_truncation_is_not_a_success_and_bad_request_is_not_retried(self):
        instance = object.__new__(api.Provider)
        instance.ledger = Mock()
        instance.args = self.args
        instance.limiter = Mock()
        instance.client = Mock()
        response = Mock()
        response.choices = [Mock(finish_reason="length")]
        instance.client.chat.completions.create.return_value = response
        with self.assertRaisesRegex(ValueError, "Incomplete/refused"):
            instance.query("question", [])
        request = httpx.Request("POST", "https://example.invalid")
        instance.client.chat.completions.create.side_effect = api.openai.BadRequestError(
            "invalid model",
            response=httpx.Response(400, request=request),
            body=None,
        )
        instance.client.chat.completions.create.reset_mock()
        with self.assertRaises(api.openai.BadRequestError):
            instance.query("question", [])
        self.assertEqual(instance.client.chat.completions.create.call_count, 1)


class ConnectionCheckTests(unittest.TestCase):
    def test_missing_key_makes_no_requests(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(check_api.openai, "OpenAI") as factory, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(check_api.check_provider("openai", [], 20))
            factory.assert_not_called()

    def test_only_metadata_methods_are_called_for_both_sdks(self):
        for provider, module, name in [
            ("openai", check_api.openai, "OpenAI"),
            ("anthropic", check_api.anthropic, "Anthropic"),
        ]:
            with patch.dict(os.environ, {
                "OPENAI_API_KEY": "fake-secret", "ANTHROPIC_API_KEY": "fake-secret",
            }), patch.object(module, name) as factory:
                client = factory.return_value.__enter__.return_value
                client.models.list.return_value.data = [Mock(id="model-one")]
                client.models.retrieve.return_value.id = "model-one"
                self.assertTrue(check_api.check_provider(provider, ["model-one"], 20))
                client.models.list.assert_called_once_with()
                client.models.retrieve.assert_called_once_with("model-one")
                self.assertEqual(
                    [call[0] for call in client.method_calls],
                    ["models.list", "models.retrieve"],
                )
                self.assertEqual(factory.call_args.kwargs["max_retries"], 0)

    def test_key_rejection_is_explicit_and_does_not_leak_secret(self):
        response = httpx.Response(
            401, request=httpx.Request("GET", "https://example.invalid"),
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-secret"}), \
                patch.object(check_api.openai, "OpenAI") as factory, \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            client = factory.return_value.__enter__.return_value
            client.models.list.side_effect = check_api.openai.AuthenticationError(
                "fake-secret", response=response, body=None,
            )
            self.assertFalse(check_api.check_provider("openai", [], 20))
        self.assertIn("HTTP 401", errors.getvalue())
        self.assertNotIn("fake-secret", errors.getvalue())

    def test_both_checks_run_when_first_fails(self):
        with patch.object(check_api, "check_provider", side_effect=[False, True]) as check:
            self.assertEqual(check_api.main(["both"]), 1)
        self.assertEqual(check.call_count, 2)


if __name__ == "__main__":
    unittest.main()
