#!/usr/bin/env python3
"""Prepare VERL SFT parquet data from data/train metadata.jsonl."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


SYSTEM_PROMPT = (
    "You are a helpful visual reasoning assistant. "
    "Answer the question about the video concisely, using only the information in the video."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/train/metadata.jsonl")
    parser.add_argument("--dataset-root", default="data/train")
    parser.add_argument("--output-dir", default="fine_tuning/data")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_rows(metadata_path: Path, dataset_root: Path) -> list[dict]:
    rows: list[dict] = []
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

            category = str(row.get("category", "")).strip()
            prompt = question if not category else f"[Category: {category}] {question}"

            # VERL MultiTurnSFTDataset expects:
            # - `messages` as list[role/content(str)]
            # - `videos` as list[path], with <video> placeholder in user content.
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"<video>\\n{prompt}"},
                {"role": "assistant", "content": answer},
            ]

            rows.append(
                {
                    "id": Path(rel_video).stem or f"sample_{index}",
                    "messages": messages,
                    # VERL expects each video entry to be a dict with a "video" key.
                    "videos": [{"video": video_path.as_uri()}],
                    "answer": answer,
                    "question": prompt,
                }
            )
    return rows


def main() -> None:
    args = parse_args()

    rows = load_rows(Path(args.metadata), Path(args.dataset_root))
    if not rows:
        raise SystemExit("No rows found. Check metadata and dataset-root paths.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    split = max(1, int(len(rows) * args.val_ratio))
    val_rows = rows[:split]
    train_rows = rows[split:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "morse_train_sft_train.parquet"
    val_path = out_dir / "morse_train_sft_val.parquet"

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    print(f"Wrote {len(train_rows)} rows -> {train_path}")
    print(f"Wrote {len(val_rows)} rows -> {val_path}")


if __name__ == "__main__":
    main()
