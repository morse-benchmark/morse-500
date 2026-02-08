#!/usr/bin/env python3

import os
import json

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

REL_ANALYSIS_DIR = os.path.join("temporal_reasoning", "analysis")
REL_TXT_DIR = "temporal_reasoning"


def safe_remove(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
        print(f"[DELETED] {path}")
    else:
        print(f"[SKIP]    {path} (not found)")


def main():
    files_to_delete = []

    print("--- Scanning files... ---")

    for model in MODELS:
        analysis_dir_path = os.path.join(model, REL_ANALYSIS_DIR)

        if not os.path.exists(analysis_dir_path):
            continue

        # Iterate over every file in the analysis directory
        for filename in os.listdir(analysis_dir_path):
            if filename.endswith(".json"):
                json_path = os.path.join(analysis_dir_path, filename)

                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Check if 'is_correct' is strictly True
                    if data.get("is_correct") is True:
                        # Infer text path
                        txt_filename = filename.replace(".json", "")
                        txt_path = os.path.join(model, REL_TXT_DIR, txt_filename)

                        # Add tuple of (json, txt) to the deletion list
                        files_to_delete.append((json_path, txt_path))
                        print(f"[FOUND]   {json_path}")

                except json.JSONDecodeError:
                    print(f"[ERROR]   Could not parse JSON: {json_path}")
                except Exception as e:
                    print(f"[ERROR]   Error reading {json_path}: {e}")

    # --- Confirmation Block ---
    count = len(files_to_delete)
    if count == 0:
        print("\nNo files found with 'is_correct': true. Exiting.")
        return

    print(f"\nFound {count} pairs of files (JSON + TXT) matching the criteria.")
    user_input = (
        input("Are you sure you want to delete these files? [y/N]: ").strip().lower()
    )

    if user_input in ["y", "yes"]:
        print("\n--- Deleting Files ---")
        for json_p, txt_p in files_to_delete:
            safe_remove(json_p)
            safe_remove(txt_p)
        print("\nCleanup complete.")
    else:
        print("\nOperation cancelled. No files were deleted.")


if __name__ == "__main__":
    main()
