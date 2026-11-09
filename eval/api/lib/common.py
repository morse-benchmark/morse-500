"""Shared benchmark inputs, checkpoints, and request rate limiting."""

import csv
import math
from pathlib import Path
import threading
import time


FIELDS = [
    "idx", "video", "ground_truth", "prediction", "question_text",
    "input_tokens", "output_tokens", "elapsed_seconds", "frame_count",
]
REPO = Path(__file__).resolve().parents[3]
DATA = REPO.parent / "datasets" / "morse-500"


class RateLimiter:
    """Space request starts across all threads, including retries."""

    def __init__(self, rpm):
        self.interval = 60 / rpm
        self.next_start = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            delay = max(0, self.next_start - now)
            if delay:
                time.sleep(delay)
            self.next_start = time.monotonic() + self.interval


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def successful(value):
    return bool(value.strip()) and not value.startswith("ERROR:")


def validate_rows(rows):
    if not rows:
        raise ValueError("Input CSV is empty")
    required = {"video", "ground_truth", "question_text"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Input CSV needs columns: {sorted(required)}")
    videos = [row["video"] for row in rows]
    if len(videos) != len(set(videos)):
        raise ValueError("Input CSV contains duplicate video names")


def load_rows(args):
    rows = read_csv(args.input)
    validate_rows(rows)
    if args.limit:
        rows = rows[:args.limit]
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} rows, found {len(rows)}")
    return rows


def add_data_arguments(parser):
    parser.add_argument("--input", type=Path, default=DATA / "test.csv")
    parser.add_argument("--video-root", type=Path, default=DATA / "test_sz512")
    parser.add_argument(
        "--frame-cache", type=Path,
        help="prepare: write reusable frames; predict: read prepared frames (never decode)",
    )
    parser.add_argument("--fps", type=float, default=4)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--limit", type=int, help="Pilot subset; use a separate output directory")
    parser.add_argument("--expected-count", type=int, help="Reject incomplete benchmark inputs")


def validate_positive_arguments(parser, args, names):
    for name in names:
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
