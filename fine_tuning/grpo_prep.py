#!/usr/bin/env python
"""Prepare GRPO data for Qwen3-VL-4B from temporal_reasoning videos.

This script pairs:
  - videos in temporal_reasoning/questions/*.mp4
  - question text in temporal_reasoning/question_text/*.txt
  - ground-truth answers in temporal_reasoning/solutions/*.txt

It writes JSONL files compatible with typical verl chat-style datasets.
Each record includes both a plain prompt/answer and a chat `messages` field
so you can adapt it to your verl config with minimal edits.
"""

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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def iter_pairs(
    video_dir: Path,
    question_dir: Path,
    solution_dir: Path,
) -> Iterable[dict]:
    for video_path in sorted(video_dir.glob("*.mp4")):
        stem = video_path.stem
        question_path = question_dir / f"{stem}.txt"
        solution_path = solution_dir / f"{stem}.txt"
        if not question_path.exists() or not solution_path.exists():
            continue
        question = read_text(question_path)
        answer = read_text(solution_path)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {"type": "text", "text": question},
                ],
            },
        ]
        yield {
            "id": stem,
            "video": str(video_path),
            "question": question,
            "answer": answer,
            "messages": messages,
        }


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-dir",
        default="/fs/nexus-scratch/mrislam/morse-500/temporal_reasoning/questions",
    )
    parser.add_argument(
        "--question-dir",
        default="/fs/nexus-scratch/mrislam/morse-500/temporal_reasoning/question_text",
    )
    parser.add_argument(
        "--solution-dir",
        default="/fs/nexus-scratch/mrislam/morse-500/temporal_reasoning/solutions",
    )
    parser.add_argument(
        "--output-dir",
        default="/fs/nexus-scratch/mrislam/morse-500/fine_tuning/data",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = list(
        iter_pairs(
            Path(args.video_dir),
            Path(args.question_dir),
            Path(args.solution_dir),
        )
    )
    if not rows:
        raise SystemExit("No pairs found. Check directories and file names.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    split = max(1, int(len(rows) * args.val_ratio))
    val_rows = rows[:split]
    train_rows = rows[split:]

    output_dir = Path(args.output_dir)
    write_jsonl(train_rows, output_dir / "temporal_grpo_train.jsonl")
    write_jsonl(val_rows, output_dir / "temporal_grpo_val.jsonl")

    print(f"Wrote {len(train_rows)} train rows and {len(val_rows)} val rows to {output_dir}")


if __name__ == "__main__":
    main()
