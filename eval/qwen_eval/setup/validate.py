"""Qwen preflight checks without model inference or network downloads."""

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import urllib.request


MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct-AWQ"
SERVER_URL = "http://127.0.0.1:8000"


def validate_setup() -> None:
    for package in ("vllm", "openai", "pandas", "tqdm"):
        print(f"{package}: {importlib.metadata.version(package)}")
    # vLLM's runtime is checked by GPU startup, not imported on CPU login nodes.
    for module in ("openai", "pandas", "tqdm"):
        importlib.import_module(module)


def validate_gpu() -> None:
    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError("No usable NVIDIA CUDA GPU; run inside a GPU allocation.")
    for index in range(torch.cuda.device_count()):
        device = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {device.name}, {device.total_memory / 1024**3:.1f} GiB")

    snapshot = Path(snapshot_download(MODEL_NAME, local_files_only=True))
    for filename in (
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "model.safetensors.index.json",
    ):
        if not (snapshot / filename).is_file():
            raise ValueError(f"Qwen cache is incomplete: missing {filename}; rerun the download job.")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map or not all(
        isinstance(name, str) and name for name in weight_map.values()
    ):
        raise ValueError("Qwen cache has an invalid safetensors weight index.")
    for filename in sorted(set(weight_map.values())):
        shard = snapshot / filename
        if not shard.is_file() or shard.stat().st_size == 0:
            raise ValueError(f"Qwen cache is incomplete: missing/empty {filename}; rerun the download job.")
    print(f"Cached Qwen snapshot: {snapshot}")


def validate_server() -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"{SERVER_URL}/health", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"Local Qwen health check returned HTTP {response.status}.")
    with opener.open(f"{SERVER_URL}/v1/models", timeout=5) as response:
        models = json.load(response)
    entries = models.get("data") if isinstance(models, dict) else None
    if not isinstance(entries, list) or not any(
        isinstance(entry, dict) and entry.get("id") == MODEL_NAME for entry in entries
    ):
        raise ValueError(f"Local server does not advertise {MODEL_NAME}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("setup", "gpu", "server"), required=True)
    args = parser.parse_args()
    try:
        validate_setup()
        if args.stage == "gpu":
            validate_gpu()
        elif args.stage == "server":
            validate_server()
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(3, f"Qwen preflight failed: {exc}\n")
    print(f"PASS: Qwen {args.stage} prerequisites.")


if __name__ == "__main__":
    main()
