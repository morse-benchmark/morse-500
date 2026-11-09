import csv
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from openai import OpenAIError


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


class ExtractionTests(unittest.TestCase):
    def run_extraction(self, predictions, answers, extra=()):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        source = Path(temporary.name) / "pred_test.csv"
        output = Path(temporary.name) / "extract_test.csv"
        rows = [
            {"video": f"abstract_reasoning_task_{index}.mp4",
             "ground_truth": truth, "prediction": prediction,
             "question_text": "Return one option letter."}
            for index, (truth, prediction) in enumerate(predictions)
        ]
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        responses = []
        for answer in answers:
            if isinstance(answer, Exception):
                responses.append(answer)
            else:
                text, finish = answer
                responses.append(SimpleNamespace(choices=[SimpleNamespace(
                    finish_reason=finish, message=SimpleNamespace(content=text)
                )]))
        create = AsyncMock(side_effect=responses)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        argv = ["extract_answers.py", str(source), "--out", str(output), *extra]
        with patch.object(sys, "argv", argv), \
                patch("openai.AsyncOpenAI", return_value=client) as factory:
            try:
                runpy.run_path(str(ROOT / "extract_answers.py"), run_name="__main__")
            except (RuntimeError, SystemExit) as exc:
                return output, create, factory, exc
        return output, create, factory, None

    def test_original_prompt_and_none_are_preserved(self):
        output, create, factory, error = self.run_extraction(
            [("None", "**D**")], [("D", "stop")]
        )
        self.assertIsNone(error)
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["api_key"], "EMPTY")
        self.assertEqual(factory.call_args.kwargs["base_url"], "http://localhost:8000/v1")
        request = create.call_args.kwargs
        self.assertEqual(request["model"], "Qwen/Qwen2.5-72B-Instruct-AWQ")
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["max_tokens"], 512)
        prompt = request["messages"][1]["content"]
        self.assertIn("Example 10", prompt)
        self.assertTrue(prompt.endswith("Model response: **D**\n\nExtracted answer:"))
        with output.open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["ground_truth"], "None")
        self.assertEqual(row["extracted_answer"], "D")

    def test_failed_predictions_remain_in_denominator(self):
        output, create, _, error = self.run_extraction(
            [("None", "ERROR: refused"), ("A", ""), ("D", "**D**")],
            [("D", "stop")], ["--allow-failed-predictions"]
        )
        self.assertIsNone(error)
        self.assertEqual(create.call_count, 1)
        with output.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["extracted_answer"] for row in rows], ["", "", "D"])
        scored = output.parent / "scores.csv"
        with patch.object(sys, "argv", [
            "plot_table.py", "--input-dir", str(output.parent), "--output", str(scored)
        ]):
            output.rename(output.with_name("extract_sz512_test.csv"))
            runpy.run_path(str(ROOT / "plot_table.py"), run_name="__main__")
        with scored.open() as stream:
            score = next(csv.DictReader(stream))
        self.assertAlmostEqual(float(score["avg_acc"]), 100 / 3)

    def test_long_ordering_answer_is_preserved_in_full(self):
        answer = "8,6,9,29,11,15,22,27,19,12,5,3,25,13,4,10,14,18,28,1,2,17,7,21,16,20,26,23,24"
        output, create, _, error = self.run_extraction(
            [(answer, answer)], [(answer, "stop")]
        )
        self.assertIsNone(error)
        self.assertEqual(create.call_args.kwargs["max_tokens"], 512)
        with output.open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["extracted_answer"], answer)
        self.assertEqual(row["ground_truth"], answer)

    def test_ground_truth_is_not_sent_to_qwen(self):
        _, create, _, error = self.run_extraction(
            [("private_ground_truth_sentinel", "The answer is D.")], [("D", "stop")]
        )
        self.assertIsNone(error)
        messages = create.call_args.kwargs["messages"]
        self.assertNotIn("private_ground_truth_sentinel", str(messages))

    def test_failed_predictions_require_explicit_policy(self):
        _, create, _, error = self.run_extraction([("A", "ERROR: length")], [])
        self.assertIsInstance(error, SystemExit)
        create.assert_not_called()

    def test_empty_or_truncated_extraction_fails(self):
        for answer, finish in [("", "stop"), ("D", "length")]:
            with self.subTest(answer=answer, finish=finish):
                output, _, _, error = self.run_extraction(
                    [("D", "**D**")], [(answer, finish)]
                )
                self.assertIsInstance(error, RuntimeError)
                self.assertTrue(output.exists())

    def test_request_error_fails_instead_of_scoring(self):
        output, _, _, error = self.run_extraction(
            [("D", "**D**")], [OpenAIError("local server unavailable")]
        )
        self.assertIsInstance(error, RuntimeError)
        self.assertTrue(output.exists())

    def test_scoring_wrapper_rejects_partial_model_files(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        results = Path(temporary.name)
        with (results / "pred_sz512_test.csv").open("w") as stream:
            stream.write("video,ground_truth,prediction,question_text\nv.mp4,D,D,q\n")
        completed = subprocess.run(
            ["bash", str(ROOT / "scripts" / "score.sh"), str(results)],
            env=dict(os.environ, PYTHON=sys.executable),
            capture_output=True, text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("expected 500 rows, found 1", completed.stderr)
        self.assertFalse((results / "benchmark_scores.csv").exists())

    def test_scoring_wrapper_preserves_500_failed_rows_and_output_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            with (results / "pred_sz512_test.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["video", "ground_truth", "question_text", "prediction"]
                )
                writer.writeheader()
                writer.writerows(
                    {"video": f"abstract_reasoning_task_{index}.mp4",
                     "ground_truth": "A", "question_text": "Return one letter.",
                     "prediction": "ERROR: original prediction failed"}
                    for index in range(500)
                )
            completed = subprocess.run(
                ["bash", str(ROOT / "scripts/score.sh"), str(results)],
                cwd=results, env=dict(os.environ, PYTHON=sys.executable),
                capture_output=True, text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (results / "extract/extract_sz512_test.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 500)
            self.assertTrue(all(row["extracted_answer"] == "" for row in rows))
            with (results / "benchmark_scores.csv").open() as stream:
                scores = list(csv.DictReader(stream))
            self.assertEqual(len(scores), 1)
            self.assertEqual(scores[0]["model"], "test")
            self.assertEqual(float(scores[0]["avg_acc"]), 0)
            self.assertEqual(list(scores[0]), [
                "model", "avg_acc", "abstract_reasoning", "mathematical_reasoning",
                "physical_reasoning", "planning_reasoning", "spatial_reasoning",
                "temporal_reasoning",
            ])

    def test_qwen_entrypoints_have_no_old_wrappers(self):
        for path in (
            "eval/extract_answers.py", "eval/plot_table.py",
            "eval/requirements-qwen.txt", "eval/download_qwen.sbatch",
            "eval/score_qwen.sbatch", "eval/test_extract_answers.py",
            "eval/api/utils/score_api.sh",
        ):
            with self.subTest(path=path):
                self.assertFalse((REPO / path).exists())
        for path in ROOT.rglob("*.sbatch"):
            self.assertEqual(path.parent, ROOT / "scripts")

    def test_qwen_job_exposes_cuda_compiler_and_ninja(self):
        script = (ROOT / "scripts" / "score_qwen.sbatch").read_text()
        setup = script.split("export OPENAI_API_KEY=EMPTY", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cuda = root / "cuda"
            python = root / "venv/bin/python"
            nvcc = cuda / "bin/nvcc"
            ninja = python.parent / "ninja"
            for path in (python, nvcc, ninja):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("#!/bin/bash\nexit 0\n")
                path.chmod(0o700)
            completed = subprocess.run(
                ["bash", "-c", setup + '\ncommand -v nvcc\ncommand -v ninja\n'],
                env=dict(os.environ, REPO=str(REPO), PYTHON=str(python),
                         CUDA_HOME=str(cuda)),
                capture_output=True, text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(str(nvcc), completed.stdout)
            self.assertIn(str(ninja), completed.stdout)
        self.assertIn("--served-model-name Qwen/Qwen2.5-72B-Instruct-AWQ", script)

    def test_qwen_job_rejects_missing_cuda_before_model_load(self):
        completed = subprocess.run(
            ["bash", str(ROOT / "setup" / "validate.sh"), "--stage", "gpu"],
            env=dict(os.environ, PYTHON=sys.executable,
                     CUDA_HOME="/nonexistent-cuda-toolkit"),
            capture_output=True, text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("CUDA compiler not found", completed.stderr)

    def test_all_slurm_logs_use_logs_directory(self):
        scripts = list(ROOT.parent.rglob("*.sbatch"))
        self.assertTrue(scripts)
        for path in scripts:
            with self.subTest(script=path):
                directives = [
                    line for line in path.read_text().splitlines()
                    if line.startswith(("#SBATCH --output=", "#SBATCH --error="))
                ]
                self.assertTrue(directives)
                self.assertTrue(all("=logs/" in line for line in directives))


if __name__ == "__main__":
    unittest.main()
