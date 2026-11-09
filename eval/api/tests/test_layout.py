import base64
import io
import json
import re
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from moviepy import ColorClip
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents import run_agents_eval
from estimates import usage
from lib import budget, common, frames
from models import run_models_eval
from utils import prepare_failed_rows
from tests.session_helper import session_env


API = Path(__file__).resolve().parents[1]
SESSION_VALUE = (r'^export (REPO|DATA_CSV|VIDEO_ROOT|VENV|RESULTS_ROOT|LOG_DIR'
                 r'|FRAME_CACHE_ROOT|AGENT_BIN)="[^"\n]+"\n')


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = dict(os.environ, PYTHONPATH="")
        self.env.pop("OPENAI_API_KEY", None)
        self.env.pop("ANTHROPIC_API_KEY", None)

    def run_script(self, script, *args):
        return subprocess.run(
            [sys.executable, str(script), *map(str, args)],
            cwd=self.root, env=self.env, capture_output=True, text=True,
        )

    def fake_python(self):
        script = self.root / "python"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['CALL_LOG'], 'a') as log:\n"
            "    log.write(json.dumps({'args': args, 'cwd': os.getcwd()}) + '\\n')\n"
            "if args and args[0] not in ('-', '-c') and not Path(args[0]).is_file():\n"
            "    sys.exit('Python script not found: ' + args[0])\n"
        )
        script.chmod(0o700)
        return script

    def test_shared_defaults_and_imports_preserve_locations(self):
        self.assertEqual(common.REPO, API.parents[1])
        self.assertEqual(common.DATA, common.REPO.parent / "datasets" / "morse-500")
        self.assertEqual(usage.PRICES, API / "estimates" / "prices.json")
        self.assertEqual(budget.PRICES, usage.PRICES)
        self.assertIs(run_agents_eval.api, common)
        self.assertIs(run_agents_eval.frame_utils, frames)

        model_args = run_models_eval.parse_args([
            "predict", "--provider", "openai", "--model", "test", "--output", "pred.csv",
        ])
        agent_args = run_agents_eval.parse_args([
            "--harness", "codex", "--model", "test", "--preflight-only",
        ])
        for args in (model_args, agent_args):
            self.assertEqual(args.input, common.DATA / "test.csv")
            self.assertEqual(args.video_root, common.DATA / "test_sz512")
        self.assertEqual(
            agent_args.frame_cache, API / "frame_cache" / "fps4_max32_sz512"
        )

    def test_evaluator_modules_use_new_filenames(self):
        self.assertEqual(Path(run_models_eval.__file__), API / "models/run_models_eval.py")
        self.assertEqual(Path(run_agents_eval.__file__), API / "agents/run_agents_eval.py")
        self.assertFalse((API / "models/api_eval.py").exists())
        self.assertFalse((API / "agents/run_agents.py").exists())

    def test_script_entrypoints_load_outside_repository(self):
        scripts = [
            "models/run_models_eval.py",
            "agents/run_agents_eval.py",
            "estimates/make_subset.py",
            "estimates/estimate_agent_costs.py",
            "estimates/estimate_costs.py",
            "estimates/static_estimate.py",
            "estimates/read_estimate.py",
            "estimates/usage.py",
            "lib/frames.py",
            "utils/check_api.py",
            "utils/prepare_failed_rows.py",
        ]
        for script in scripts:
            with self.subTest(script=script):
                result = self.run_script(API / script, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_shared_and_model_prepare_cli_use_compatible_frame_caches(self):
        video = self.root / "clip.mp4"
        clip = ColorClip(size=(32, 16), color=(80, 110, 40)).with_duration(1)
        try:
            clip.write_videofile(str(video), fps=4, codec="libx264", audio=False, logger=None)
        finally:
            clip.close()

        source = self.root / "input.csv"
        rows = [{"video": video.name, "ground_truth": "A", "question_text": "question"}]
        common.save_csv(source, rows, list(rows[0]))
        shared_cache = self.root / "shared-cache"
        model_cache = self.root / "model-cache"
        options = [
            "--input", source, "--video-root", self.root,
            "--fps", "4", "--max-frames", "3", "--max-side", "16",
            "--expected-count", "1",
        ]
        for script, cache in (
            (frames.__file__, shared_cache),
            (run_models_eval.__file__, model_cache),
            (run_models_eval.__file__, shared_cache),
        ):
            result = self.run_script(script, "prepare", *options, "--frame-cache", cache)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Cache verified 1/1", result.stdout)

        args = run_models_eval.parse_args([
            "prepare", *map(str, options), "--frame-cache", str(shared_cache),
        ])
        shared_images = run_agents_eval.frame_utils.load_cached_frames(args, video.name)
        args.frame_cache = model_cache
        self.assertEqual(shared_images, frames.load_cached_frames(args, video.name))
        self.assertEqual(len(shared_images), 3)
        for encoded in shared_images:
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                self.assertEqual(image.size, (16, 8))

        missing_cache = self.root / "missing-cache"
        result = self.run_script(
            frames.__file__, "prepare", "--input", source, "--video-root", self.root,
            "--frame-cache", missing_cache, "--expected-count", "2",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Expected 2 rows, found 1", result.stderr)
        self.assertFalse(missing_cache.exists())

    def test_recovery_utility_runs_from_outside_repository(self):
        source = self.root / "source.csv"
        predictions = self.root / "predictions.csv"
        output = self.root / "failed.csv"
        rows = [
            {"id": str(index), "video": f"{index}.mp4", "ground_truth": "A", "question_text": "q"}
            for index in (11, 22, 33)
        ]
        common.save_csv(source, rows, list(rows[0]))
        common.save_csv(predictions, [
            {"video": "11.mp4", "prediction": "answer"},
            {"video": "22.mp4", "prediction": "ERROR: failed"},
        ], ["video", "prediction"])

        result = self.run_script(
            prepare_failed_rows.__file__, "--input", source,
            "--predictions", predictions, "--output", output,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        failed = common.read_csv(output)
        self.assertEqual([row["video"] for row in failed], ["22.mp4", "33.mp4"])
        self.assertEqual([row["idx"] for row in failed], ["22", "33"])

        result = subprocess.run(
            [sys.executable, "-S", prepare_failed_rows.__file__, "--help"],
            cwd=self.root, env=self.env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_usage_cli_finds_shared_default_prices(self):
        ledger = self.root / "prediction.usage.jsonl"
        ledger.write_text(json.dumps({
            "event_id": "event", "provider": "openai", "model": "gpt-6-astra",
            "stage": "predict", "usage": {"input_tokens": 1000, "output_tokens": 100},
        }) + "\n")
        output = self.root / "summary.json"
        result = self.run_script(usage.__file__, self.root, "--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        prices = json.loads(usage.PRICES.read_text())
        self.assertEqual(report["models"], usage.summarize([ledger], prices))
        self.assertEqual(report["price_as_of"], prices["as_of"])
        self.assertEqual(json.loads(output.read_text()), report)

    def test_connectivity_batch_invokes_relocated_utility(self):
        log = self.root / "check-calls.jsonl"
        env = session_env(self.root, PYTHONPATH="", REPO=str(common.REPO),
                          PYTHON=str(self.fake_python()), CALL_LOG=str(log))
        script = API / "scripts" / "extras" / "check_api.sbatch"
        for args in ([], ["anthropic", "--model", "model"]):
            subprocess.run(["bash", str(script), *args], env=env, check=True, capture_output=True)
        calls = [json.loads(line)["args"] for line in log.read_text().splitlines()]
        self.assertEqual(calls, [
            [str(API / "utils/check_api.py"), "both"],
            [str(API / "utils/check_api.py"), "anthropic", "--model", "model"],
        ])

    def test_scoring_wrapper_finds_repository_from_new_location(self):
        log = self.root / "score-calls.jsonl"
        results = self.root / "results"
        results.mkdir()
        (results / "pred_sz512_test.csv").write_text(
            "video,ground_truth,prediction,question_text\n" + "v.mp4,A,A,q\n" * 500
        )
        env = dict(self.env, PYTHON=str(self.fake_python()), CALL_LOG=str(log))
        result = subprocess.run(
            ["bash", str(common.REPO / "eval/qwen_eval/scripts/score.sh"), str(results)],
            cwd=self.root, env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual([call["args"][0] for call in calls], [
            "-", "eval/qwen_eval/extract_answers.py", "eval/qwen_eval/plot_table.py",
        ])
        self.assertTrue(all(call["cwd"] == str(common.REPO) for call in calls))

    def test_entrypoints_fail_fast_without_session(self):
        env = {key: value for key, value in self.env.items() if key != "MORSE_SESSION_ENV"}
        for script in ("scripts/run_local.sh", "scripts/estimate_local.sh",
                       "scripts/extras/retry_failed.sh", "scripts/submit_slurm.sh",
                       "setup/validate.sh", "setup/setup_env.sh", "setup/install_harnesses.sh"):
            result = subprocess.run(["bash", str(API / script)], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 3, script)
            self.assertIn("SESSION NOT CONFIGURED", result.stderr, script)
        env = session_env(self.root, VIDEO_ROOT=str(self.root / "missing"))
        result = subprocess.run(["bash", str(API / "scripts/run_local.sh"), "model", "openai"],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 3)
        self.assertIn("VIDEO_ROOT directory does not exist", result.stderr)

    def session_tree(self, **values):
        repo = self.root / "repo"
        for name in ("setup/session_env.sh", "lib/session.sh"):
            target = repo / "eval/api" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            text = (API / name).read_text()
            if name == "setup/session_env.sh":
                # Drop a developer's local values so tests start from the shipped template.
                text = re.sub(SESSION_VALUE, "", text, flags=re.MULTILINE)
            target.write_text(text)
        data = self.root / "datasets/morse-500"
        (data / "test_sz512").mkdir(parents=True, exist_ok=True)
        (data / "test.csv").touch()
        return repo

    def fill_session(self, repo, **overrides):
        data = self.root / "datasets/morse-500"
        values = {
            "REPO": repo, "DATA_CSV": data / "test.csv", "VIDEO_ROOT": data / "test_sz512",
            "VENV": repo / ".morse_venv", "RESULTS_ROOT": repo / "eval/results",
            "LOG_DIR": repo / "logs", "FRAME_CACHE_ROOT": repo / "eval/api/frame_cache",
            "AGENT_BIN": self.root / "agent-bin",
        }
        values.update(overrides)
        target = repo / "eval/api/setup/session_env.sh"
        text = target.read_text()
        for name, value in values.items():
            placeholder = f'export {name}=""\n'
            self.assertEqual(text.count(placeholder), 1, name)
            text = text.replace(placeholder, f'export {name}="{value}"\n')
        target.write_text(text)

    def source_session(self, repo, keys):
        script = (
            "source eval/api/setup/session_env.sh; status=$?; "
            "env | grep -E '^(MORSE_SESSION_ENV|DATA_CSV|RESULTS_ROOT|LOG_DIR|AGENT_BIN"
            "|OPENAI_API_KEY|ANTHROPIC_API_KEY)='; exit $status"
        )
        env = {key: value for key, value in self.env.items()
               if key not in ("MORSE_SESSION_ENV", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
        env["HOME"] = str(self.root / "home")
        return subprocess.run(["bash", "-c", script], cwd=repo, env=env, input=keys,
                              capture_output=True, text=True)

    def test_session_setup_exports_paths_and_silently_read_keys(self):
        repo = self.session_tree()
        self.fill_session(repo)
        result = self.source_session(repo, "sk-openai\nsk-anthropic\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        exported = dict(line.split("=", 1) for line in result.stdout.splitlines()
                        if not line.startswith("MORSE session ready"))
        self.assertEqual(exported["MORSE_SESSION_ENV"], "1")
        self.assertEqual(exported["OPENAI_API_KEY"], "sk-openai")
        self.assertEqual(exported["ANTHROPIC_API_KEY"], "sk-anthropic")
        self.assertEqual(Path(exported["RESULTS_ROOT"]), repo / "eval/results")
        for name in ("RESULTS_ROOT", "LOG_DIR", "AGENT_BIN"):
            self.assertTrue(Path(exported[name]).is_dir(), name)
        self.assertNotIn("sk-", result.stderr)

    def test_session_setup_has_no_default_paths(self):
        repo = self.session_tree()
        result = self.source_session(repo, "sk-openai\nsk-anthropic\n")
        self.assertEqual(result.returncode, 1)
        for name in ("REPO", "DATA_CSV", "VIDEO_ROOT", "VENV", "RESULTS_ROOT",
                     "LOG_DIR", "FRAME_CACHE_ROOT", "AGENT_BIN"):
            self.assertIn(f"{name} is not filled in", result.stderr)
        self.assertIn("fill in the review block", result.stderr)
        self.assertNotIn("API key", result.stderr)
        self.assertNotIn("MORSE_SESSION_ENV", result.stdout)
        self.assertNotIn("API_KEY", result.stdout)
        for name in ("DATA_CSV", "RESULTS_ROOT", "LOG_DIR", "AGENT_BIN"):
            self.assertIn(f"{name}=\n", result.stdout)
        self.assertFalse((repo / "eval/results").exists())
        # Values inherited from an earlier session are not used as fallbacks.
        env = session_env(self.root, REPO=str(repo))
        result = subprocess.run(["bash", "-c", "source eval/api/setup/session_env.sh"],
                                cwd=repo, env=env, input="a\nb\n",
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("REPO is not filled in", result.stderr)

    def test_session_setup_fails_on_missing_paths_or_keys(self):
        repo = self.session_tree()
        self.fill_session(repo)
        result = self.source_session(repo, "sk-openai\n\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("both OPENAI_API_KEY and ANTHROPIC_API_KEY", result.stderr)
        self.assertNotIn("MORSE_SESSION_ENV", result.stdout)
        (self.root / "datasets/morse-500/test.csv").unlink()
        result = self.source_session(repo, "sk-openai\nsk-anthropic\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("DATA_CSV file does not exist", result.stderr)
        self.assertNotIn("OPENAI_API_KEY", result.stdout)
        result = subprocess.run(["bash", str(API / "setup/session_env.sh")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Source this file", result.stderr)

    def test_session_setup_rejects_relative_or_misparented_paths(self):
        repo = self.session_tree()
        self.fill_session(repo, LOG_DIR="logs",
                          RESULTS_ROOT=self.root / "typo/eval/results")
        result = self.source_session(repo, "sk-openai\nsk-anthropic\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("LOG_DIR must be an absolute path: logs", result.stderr)
        self.assertIn("RESULTS_ROOT parent directory does not exist", result.stderr)
        self.assertFalse((self.root / "typo").exists())
        self.assertNotIn("MORSE_SESSION_ENV", result.stdout)
        self.assertNotIn("API_KEY", result.stdout)

    def test_setup_env_installs_into_session_venv_and_reruns(self):
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        uv = bin_dir / "uv"
        uv.write_text(
            "#!/usr/bin/env bash\n"
            "echo \"$*\" >> \"$CALL_LOG\"\n"
            "if [[ $1 == venv ]]; then\n"
            "  [[ ! -e $2 ]] || { echo 'venv exists' >&2; exit 2; }\n"
            "  mkdir -p \"$2/bin\"; printf '#!/bin/sh\\n' > \"$2/bin/python\"; chmod +x \"$2/bin/python\"\n"
            "fi\n"
        )
        uv.chmod(0o700)
        log = self.root / "uv-calls.log"
        venv = self.root / "venv"
        env = session_env(self.root, VENV=str(venv), PYTHON=str(venv / "bin/python"),
                          CALL_LOG=str(log), PATH=f"{bin_dir}:{os.environ['PATH']}")
        for _ in range(2):
            result = subprocess.run(["bash", str(API / "setup/setup_env.sh")], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        calls = log.read_text().splitlines()
        self.assertEqual(calls[0], f"venv {venv}")
        self.assertEqual([c for c in calls if c.startswith("venv")], [f"venv {venv}"])
        self.assertEqual(sum(c.startswith(f"pip install --python {venv}/bin/python") for c in calls), 2)
        self.assertIn("Reusing existing virtual environment", result.stdout)

    def test_scripts_reject_relative_session_paths(self):
        env = session_env(self.root, LOG_DIR="logs")
        result = subprocess.run(["bash", str(API / "scripts/run_local.sh"), "model", "openai"],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 3)
        self.assertIn("LOG_DIR must be an absolute path", result.stderr)

    def test_all_sbatch_entrypoints_live_in_scripts(self):
        self.assertTrue(list((API / "scripts").glob("*.sbatch")))
        for path in API.rglob("*.sbatch"):
            self.assertIn(path.parent, (API / "scripts", API / "scripts/extras"), path)
        for removed in (
            API / "agents/run_agents.sbatch", API / "models/run_api.sbatch",
            API / "models/run_failed_rows.sbatch", API / "estimates/agent_estimates.sbatch",
            API / "utils/check_api.sbatch",
        ):
            self.assertFalse(removed.exists(), removed)


if __name__ == "__main__":
    unittest.main()
