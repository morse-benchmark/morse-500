#!/usr/bin/env python3
"""
tabulate_summary_percentages.py

Hardcoded version:
  - Uses a fixed list of Qwen3-VL models (NO *_analyze)
  - Each model directory must contain: summary.json
  - Produces a primary error percentage table
  - Adds an 'accuracy' column from summary.json

Percentages are computed as (% of total questions):
  pct(error_type) = count(error_type) / total_questions * 100
"""

import argparse
import csv
import json
import os
from typing import Dict, List


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

CATEGORY_NAME = "temporal_reasoning"


def load_summary(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_percentages(counts: Dict[str, int], total: int) -> Dict[str, float]:
    if total <= 0:
        return {k: 0.0 for k in counts}
    return {k: (int(v) / total) * 100.0 for k, v in counts.items()}


def write_csv(path: str, header: List[str], rows: List[List[str]]) -> None:
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def fmt(x: float, digits: int) -> str:
    return f"{x:.{digits}f}"


def print_table(title: str, header: List[str], rows: List[List[str]]) -> None:
    print("\n" + title)
    print("-" * len(title))

    widths = [len(h) for h in header]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def row_fmt(r):
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(r))

    print(row_fmt(header))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(row_fmt(r))


def normalize_accuracy(acc) -> float:
    """Return accuracy as percent in [0, 100]."""
    if acc is None:
        return 0.0
    try:
        acc = float(acc)
    except Exception:
        return 0.0
    return acc * 100.0 if acc <= 1.0 else acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="tables")
    parser.add_argument("--digits", type=int, default=2)
    args = parser.parse_args()

    # First pass: load each model summary, store counts + totals + accuracy,
    # and collect the union of all primary categories for stable columns.
    all_primary_categories = set()
    primary_counts_per_model: Dict[str, Dict[str, int]] = {}
    total_questions_per_model: Dict[str, int] = {}
    accuracy_per_model: Dict[str, float] = {}

    for model in MODELS:
        summary_path = os.path.join(
            model, CATEGORY_NAME, "analysis", "summary_new.json"
        )
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Missing: {summary_path}")

        data = load_summary(summary_path)

        total_q = int(data["total"])  # total questions
        total_questions_per_model[model] = total_q

        accuracy_per_model[model] = normalize_accuracy(data.get("accuracy"))

        counts = {str(k): int(v) for k, v in data["category distribution"].items()}
        if "None" in counts:
            del counts["None"]
        primary_counts_per_model[model] = counts

        all_primary_categories.update(counts.keys())

    # Build table columns once (stable)
    cats = sorted(all_primary_categories)
    header = ["model", "accuracy", "total_questions"] + cats

    # Second pass: build rows
    rows: List[List[str]] = []
    for model in MODELS:
        counts = primary_counts_per_model[model]
        total_q = total_questions_per_model[model]
        pcts = compute_percentages(counts, total_q)

        row = [
            model,
            fmt(accuracy_per_model[model], args.digits),
            str(total_q),
        ]
        for c in cats:
            row.append(fmt(pcts.get(c, 0.0), args.digits))
        rows.append(row)

    # Output
    print_table("Primary Error Percentages (of total questions)", header, rows)
    write_csv(
        os.path.join(args.outdir, f"{CATEGORY_NAME}_primary_percentages.csv"),
        header,
        rows,
    )
    print(f"\n[OK] Tables written to: {args.outdir}/")


if __name__ == "__main__":
    main()
