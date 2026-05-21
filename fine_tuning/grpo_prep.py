#!/usr/bin/env python3
"""Prepare VERL GRPO data for Qwen3-VL-4B from data/train metadata.jsonl."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


SYSTEM_PROMPT = (
    "You are a helpful visual reasoning assistant. Answer the question "
    "about the video concisely, using only the information in the video."
)


def iter_rows(metadata_path: Path, dataset_root: Path) -> Iterable[dict]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue

            row = json.loads(line)
            rel_video = str(row.get("file_name", "")).strip()
            question = str(row.get("question_text", "")).strip()
            answer = str(row.get("solution", "")).strip()
            if not rel_video or not question or not answer:
                continue

            video_path = (dataset_root / rel_video).resolve()
            if not video_path.exists():
                continue

            sample_id = Path(rel_video).stem or f"sample_{index}"
            category = str(row.get("category", "")).strip()
            prompt_text = question if not category else f"[Category: {category}] {question}"

            reasoning_trace = str(row.get("reasoning_trace", "")).strip()
            extra = {"reasoning_trace": reasoning_trace} if reasoning_trace else {}

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(video_path)},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]

            yield {
                "uid": sample_id,
                "prompt": messages,
                "answer": answer,
                "data_source": category or "morse500",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    "question": prompt_text,
                    "video": str(video_path),
                    **extra,
                },
            }


def write_json(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/train/metadata.jsonl")
    parser.add_argument("--dataset-root", default="data/train")
    parser.add_argument("--output-dir", default="fine_tuning/data")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = list(iter_rows(Path(args.metadata), Path(args.dataset_root)))
    if not rows:
        raise SystemExit("No training rows found. Check metadata and dataset-root paths.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    split = max(1, int(len(rows) * args.val_ratio))
    val_rows = rows[:split]
    train_rows = rows[split:]

    output_dir = Path(args.output_dir)
    write_json(train_rows, output_dir / "morse_train_grpo_train.json")
    write_json(val_rows, output_dir / "morse_train_grpo_val.json")

    print(f"Wrote {len(train_rows)} train rows and {len(val_rows)} val rows to {output_dir}")


if __name__ == "__main__":
    main()
