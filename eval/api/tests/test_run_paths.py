import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import run_paths
from agents import run_agents_eval
from models import run_models_eval
from tests.session_helper import session_env


API = Path(__file__).resolve().parents[1]


class RunPathsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results = self.root / "eval/results"

    def test_fresh_resume_and_usage_directory(self):
        with patch.object(run_paths, "RESULTS", self.results):
            directory = run_paths.initialize(name="pilot")
            self.assertTrue((directory / "run_usage").is_dir())
            checkpoint = directory / "pred.csv"
            checkpoint.write_text("retained")
            self.assertEqual(run_paths.initialize(resume="pilot"), directory)
            self.assertEqual(checkpoint.read_text(), "retained")
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_paths.initialize(name="pilot")
            with self.assertRaisesRegex(ValueError, "does not exist"):
                run_paths.initialize(resume="missing")
            self.assertEqual(
                run_paths.usage_path(directory / "estimate/pred.csv"),
                directory / "run_usage/estimate/pred.usage.jsonl",
            )

    def test_generated_runs_are_distinct_and_outside_api(self):
        first = run_paths.initialize(root=self.results, prefix="model")
        second = run_paths.initialize(root=self.results, prefix="agent")
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, self.results)
        self.assertEqual(run_paths.RESULTS, API.parent / "results")

    def test_invalid_and_conflicting_paths_are_rejected(self):
        for kwargs in (
            {"name": "../escape"}, {"name": "one", "resume": "two"},
            {"output": self.root / "api/results/run"},
            {"name": "one", "output": self.results / "two"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_paths.initialize(root=self.results, **kwargs)
        self.results.mkdir(parents=True)
        (self.results / "external").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            run_paths.initialize(root=self.results, resume="external")

    def test_low_level_evaluators_initialize_and_resume_explicitly(self):
        with patch.object(run_paths, "RESULTS", self.results):
            for evaluator, arguments, prefix in (
                (run_models_eval, ["predict", "--provider", "openai"], "model-openai"),
                (run_agents_eval, ["--harness", "claude"], "agent-claude"),
            ):
                args = evaluator.parse_args([
                    *arguments, "--model", "test", "--run-name", prefix,
                    "--budget-usd", "1",
                ])
                run_paths.initialize_evaluator(args, prefix)
                self.assertTrue((args.output.parent / "run_usage").is_dir())
                self.assertEqual(args.budget_file, args.output.parent / "run_usage/budget.jsonl")
                self.assertTrue(args.output.name.startswith("pred_"))
                output = args.output
                resumed = evaluator.parse_args([
                    *arguments, "--model", "test", "--resume-run", prefix,
                    "--output", str(output),
                ])
                run_paths.initialize_evaluator(resumed, prefix)
                self.assertEqual(resumed.output, output)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    run_paths.initialize_evaluator(args, prefix)

    def test_bootstrap_pilot_and_full_evaluation_share_one_run(self):
        python = self.root / "python"
        log = self.root / "calls"
        python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['CALL_LOG'], 'a') as stream:\n"
            " stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if sys.argv[1].endswith('lib/run_paths.py'):\n"
            " os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        )
        python.chmod(0o700)
        source = self.root / "test.csv"
        source.touch()
        env = session_env(
            self.root, PYTHON=str(python),
            DATA_CSV=str(source), VIDEO_ROOT=str(self.root), OPENAI_API_KEY="test",
            RUN_NAME="bootstrap", ESTIMATE="bootstrap", CALL_LOG=str(log),
        )
        for key in ("RESUME_RUN", "OUTPUT_DIR", "PREFLIGHT_ONLY", "BUDGET_USD", "BUDGET_FILE",
                    "ESTIMATE_DIR", "BUDGET_SOURCE", "BUDGET_POLICY", "ACTION"):
            env.pop(key, None)
        subprocess.run(
            ["bash", str(API / "lib/run_eval.sh"), "model", "openai", "test"],
            env=env, check=True, capture_output=True, text=True,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(sum(call[0].endswith("lib/run_paths.py") for call in calls), 1)
        predictions = [call for call in calls if call[0].endswith("models/run_models_eval.py")]
        self.assertEqual(len(predictions), 2)
        directory = self.results / "bootstrap"
        self.assertEqual(
            [Path(call[call.index("--output") + 1]).parent for call in predictions],
            [directory / "estimate", directory],
        )
        self.assertTrue(all(call[call.index("--resume-run") + 1] == "bootstrap"
                            for call in predictions))

    def test_wrappers_initialize_before_preflight_for_every_mode_and_action(self):
        python = self.root / "python"
        python.write_text(
            f"#!{sys.executable}\n"
            "import os, sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1].endswith('lib/run_paths.py'):\n"
            " os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
            "if sys.argv[1].endswith('utils/check_api.py'):\n"
            " assert (Path(os.environ['EXPECTED_RUN']) / 'run_usage').is_dir()\n"
            " sys.exit(19)\n"
            "if '--preflight-only' in sys.argv:\n"
            " assert (Path(os.environ['EXPECTED_RUN']) / 'run_usage').is_dir()\n"
            " sys.exit(19)\n"
        )
        python.chmod(0o700)
        tools = self.root / "bin"
        tools.mkdir()
        for tool in ("bwrap", "socat", "codex", "claude"):
            path = tools / tool
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o700)
        source = self.root / "test.csv"
        source.touch()
        for mode in ("model", "agent"):
            for action in ("run", "estimate"):
                name = f"{mode}-{action}"
                directory = self.results / name
                env = session_env(
                    self.root, PYTHON=str(python),
                    DATA_CSV=str(source), VIDEO_ROOT=str(self.root),
                    OPENAI_API_KEY="test", AGENT_BIN=str(tools),
                    CODEX_BINARY=str(tools / "codex"), CLAUDE_BINARY=str(tools / "claude"),
                    PATH=f"{tools}:{os.environ['PATH']}", ACTION=action,
                    RUN_NAME=name, EXPECTED_RUN=str(directory),
                )
                for key in ("RESUME_RUN", "OUTPUT_DIR", "PREFLIGHT_ONLY", "BUDGET_USD"):
                    env.pop(key, None)
                command = ["bash", str(API / "lib/run_eval.sh"), mode, "openai", "test"]
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 3, result.stderr)
                self.assertTrue((directory / "run_usage").is_dir())
                self.assertIn("preflight failed", result.stderr.lower())
                env["RESUME_RUN"] = env.pop("RUN_NAME")
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertNotIn("initialization failed", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
