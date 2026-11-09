import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen_preflight", ROOT / "setup/validate.py")
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)


class PreflightTests(unittest.TestCase):
    def test_setup_does_not_import_vllm_on_cpu(self):
        with patch.object(validate.importlib.metadata, "version", return_value="test") as version, \
                patch.object(validate.importlib, "import_module") as imported:
            validate.validate_setup()
        self.assertEqual([call.args[0] for call in version.call_args_list],
                         ["vllm", "openai", "pandas", "tqdm"])
        self.assertEqual([call.args[0] for call in imported.call_args_list],
                         ["openai", "pandas", "tqdm"])

    def test_setup_rejects_missing_dependency(self):
        with patch.object(validate.importlib.metadata, "version",
                          side_effect=validate.importlib.metadata.PackageNotFoundError("vllm")):
            with self.assertRaises(ImportError):
                validate.validate_setup()

    def gpu_modules(self, snapshot, available=True):
        return {
            "torch": SimpleNamespace(cuda=SimpleNamespace(
                is_available=lambda: available,
                device_count=lambda: 1,
                get_device_properties=lambda index: SimpleNamespace(
                    name="mock H200", total_memory=141 * 1024**3,
                ),
            )),
            "huggingface_hub": SimpleNamespace(
                snapshot_download=Mock(return_value=str(snapshot)),
            ),
        }

    def create_snapshot(self, root):
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (root / filename).write_text("{}")
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"layer": "model-00001-of-00001.safetensors"},
        }))

    def test_gpu_rejects_cpu_before_reading_cache(self):
        modules = self.gpu_modules("/unused", available=False)
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "GPU allocation"):
                validate.validate_gpu()
        modules["huggingface_hub"].snapshot_download.assert_not_called()

    def test_gpu_rejects_partial_download_and_accepts_complete_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_snapshot(root)
            modules = self.gpu_modules(root)
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(ValueError, "missing/empty model-"):
                    validate.validate_gpu()
                (root / "model-00001-of-00001.safetensors").write_bytes(b"mock weights")
                validate.validate_gpu()
            for call in modules["huggingface_hub"].snapshot_download.call_args_list:
                self.assertEqual(call.args, (validate.MODEL_NAME,))
                self.assertEqual(call.kwargs, {"local_files_only": True})

    def test_gpu_rejects_invalid_weight_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_snapshot(root)
            (root / "model.safetensors.index.json").write_text('{"weight_map": {}}')
            with patch.dict(sys.modules, self.gpu_modules(root)):
                with self.assertRaisesRegex(ValueError, "invalid safetensors"):
                    validate.validate_gpu()

    def response(self, data, status=200):
        stream = io.BytesIO(json.dumps(data).encode())
        stream.status = status
        return stream

    def test_server_checks_loopback_health_and_model_without_proxies(self):
        opener = Mock()
        opener.open.side_effect = [
            self.response({}),
            self.response({"data": [{"id": validate.MODEL_NAME}]}),
        ]
        with patch.object(validate.urllib.request, "build_opener", return_value=opener) as factory:
            validate.validate_server()
        self.assertEqual(factory.call_args.args[0].proxies, {})
        self.assertEqual([call.args[0] for call in opener.open.call_args_list], [
            "http://127.0.0.1:8000/health", "http://127.0.0.1:8000/v1/models",
        ])
        self.assertTrue(all(call.kwargs == {"timeout": 5} for call in opener.open.call_args_list))

    def test_server_rejects_wrong_model_and_unhealthy_server(self):
        opener = Mock()
        with patch.object(validate.urllib.request, "build_opener", return_value=opener):
            opener.open.side_effect = [
                self.response({}), self.response({"data": [{"id": "other-model"}]}),
            ]
            with self.assertRaisesRegex(ValueError, "does not advertise"):
                validate.validate_server()
            opener.open.side_effect = [self.response({}, status=503)]
            with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                validate.validate_server()

    def test_failed_server_preflight_exits_nonzero_with_error(self):
        with patch.object(sys, "argv", ["validate.py", "--stage", "server"]), \
                patch.object(validate, "validate_setup"), \
                patch.object(validate, "validate_server", side_effect=URLError("unavailable")), \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as failure:
                validate.main()
        self.assertEqual(failure.exception.code, 3)
        self.assertIn("Qwen preflight failed", stderr.getvalue())
        self.assertIn("unavailable", stderr.getvalue())

    def test_setup_script_uses_separate_environment_and_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uv = root / "uv"
            uv.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n')
            uv.chmod(0o700)
            log = root / "calls"
            venv = root / "qwen-venv"
            result = subprocess.run(
                ["bash", str(ROOT / "setup/setup_env.sh")], cwd=root,
                env=dict(os.environ, PATH=f"{root}:{os.environ['PATH']}",
                         VENV=str(venv), CALL_LOG=str(log)),
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(log.read_text().splitlines(), [
                f"venv --python 3.12 {venv}",
                f"pip install --python {venv}/bin/python -r {ROOT}/requirements.txt",
            ])

    def test_validation_rejects_invalid_stage_and_missing_python(self):
        for args, expected, message in (
            (["--stage", "unknown"], 2, "Usage:"),
            (["--stage"], 2, "Usage:"),
            (["--stage", "setup"], 3, "Python interpreter missing"),
        ):
            with self.subTest(args=args):
                result = subprocess.run(
                    ["bash", str(ROOT / "setup/validate.sh"), *args],
                    env=dict(os.environ, PYTHON="/nonexistent-qwen-python"),
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, expected)
                self.assertIn(message, result.stderr)

    def test_gpu_job_uses_new_preflights_and_cleans_up_its_server(self):
        for failed_stage in ("", "gpu", "server"):
            with self.subTest(failed_stage=failed_stage), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bin_dir = root / "bin"
                bin_dir.mkdir()
                cuda_bin = root / "cuda/bin"
                cuda_bin.mkdir(parents=True)
                log = root / "calls.jsonl"
                ready = root / "ready"
                python = bin_dir / "python"
                python.write_text(
                    f"#!{sys.executable}\n"
                    "import json, os, signal, sys, time\n"
                    "from pathlib import Path\n"
                    "def record(args):\n"
                    "    with open(os.environ['CALL_LOG'], 'a') as stream:\n"
                    "        stream.write(json.dumps(args) + '\\n')\n"
                    "args = sys.argv[1:]\n"
                    "record(args)\n"
                    "if args[:2] == ['-m', 'vllm.entrypoints.cli.main']:\n"
                    "    def stop(signum, frame):\n"
                    "        record(['server-stop'])\n"
                    "        sys.exit(0)\n"
                    "    signal.signal(signal.SIGTERM, stop)\n"
                    "    Path(os.environ['READY']).write_text('ready')\n"
                    "    time.sleep(30)\n"
                    "elif args and args[0].endswith('/setup/validate.py'):\n"
                    "    if os.environ.get('FAIL_STAGE') == args[-1]:\n"
                    "        sys.exit('mock preflight failure')\n"
                )
                python.chmod(0o700)
                for path in (cuda_bin / "nvcc", bin_dir / "ninja"):
                    path.write_text("#!/bin/bash\nexit 0\n")
                    path.chmod(0o700)
                curl = bin_dir / "curl"
                curl.write_text(
                    '#!/bin/bash\n'
                    'for (( attempt=0; attempt<1000; attempt++ )); do\n'
                    '    [[ ! -f "$READY" ]] || exit 0\n'
                    '    /bin/sleep 0.01\n'
                    'done\n'
                    'exit 1\n'
                )
                curl.chmod(0o700)
                results = root / "results"
                results.mkdir()
                (results / "pred_sz512_test.csv").write_text(
                    "video,ground_truth,question_text,prediction\n" + "v.mp4,A,q,A\n" * 500
                )
                completed = subprocess.run(
                    ["bash", str(ROOT / "scripts/score_qwen.sbatch"), str(results)],
                    cwd=root, capture_output=True, text=True, timeout=45,
                    env=dict(os.environ, REPO=str(ROOT.parents[1]), PYTHON=str(python),
                             CUDA_HOME=str(root / "cuda"), HF_HOME=str(root / "cache"),
                             CALL_LOG=str(log), READY=str(ready), FAIL_STAGE=failed_stage),
                )
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                preflights = [
                    call[-1] for call in calls
                    if call and call[0].endswith("/setup/validate.py")
                ]
                scored = any(call and call[0] == "eval/qwen_eval/plot_table.py" for call in calls)
                self.assertEqual(preflights, ["gpu"] if failed_stage == "gpu" else ["gpu", "server"])
                self.assertEqual(completed.returncode == 0, not failed_stage, completed.stderr)
                self.assertEqual(scored, not failed_stage)
                if failed_stage == "gpu":
                    self.assertFalse(ready.exists())
                else:
                    self.assertIn(["server-stop"], calls)
                    self.assertTrue(any(call[:2] == ["-m", "vllm.entrypoints.cli.main"] for call in calls))


if __name__ == "__main__":
    unittest.main()
