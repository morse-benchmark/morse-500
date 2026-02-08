#!/usr/bin/env python3

import os

MODELS = [
    "Qwen3-VL-2B-Instruct",
    "Qwen3-VL-2B-Thinking",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-4B-Thinking",
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-8B-Thinking",
    "Qwen3-VL-30B-A3B-Instruct-FP8",
    "Qwen3-VL-30B-A3B-Thinking-FP8",
    "Qwen3-VL-32B-Instruct-FP8",
    "Qwen3-VL-32B-Thinking-FP8",
    "Qwen3-VL-235B-A22B-Instruct-FP8",
    "Qwen3-VL-235B-A22B-Thinking-FP8",
]

TXT_REL_PATH = os.path.join(
    "temporal_reasoning",
    "duration2d_n2_seed2813.txt",
)

JSON_REL_PATH = os.path.join(
    "temporal_reasoning",
    "analysis",
    "duration2d_n2_seed2813.txt.json",
)


def remove_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
        print(f"[REMOVED] {path}")
    else:
        print(f"[SKIP]    {path} (not found)")


def main():
    for model in MODELS:
        txt_path = os.path.join(model, TXT_REL_PATH)
        json_path = os.path.join(model, JSON_REL_PATH)

        remove_file(txt_path)
        remove_file(json_path)


if __name__ == "__main__":
    main()
