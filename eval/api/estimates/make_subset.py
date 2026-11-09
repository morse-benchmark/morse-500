"""Write a deterministic, category-stratified MORSE-500 subset CSV.

Equal allocation: N // K examples per category; any remainder goes one each to
the largest categories (ties broken by name). Rows keep benchmark order and
columns. Re-running with the same arguments reproduces identical bytes, so
run_agents_eval.py resume metadata (input SHA-256) stays valid.
"""

import argparse
from collections import defaultdict
import csv
import io
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import common as api


def allocate(sizes, total):
    if total < len(sizes):
        raise ValueError(f"Need at least one example per category ({len(sizes)}); got {total}")
    if total > sum(sizes.values()):
        raise ValueError("Subset is larger than the benchmark")
    base, remainder = divmod(total, len(sizes))
    order = sorted(sizes, key=lambda name: (-sizes[name], name))
    counts = {name: base + (1 if rank < remainder else 0) for rank, name in enumerate(order)}
    for name, count in counts.items():
        if count > sizes[name]:
            raise ValueError(f"Category {name} has only {sizes[name]} rows; needs {count}")
    return counts


def stratified_subset(rows, total, seed, column="category"):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if not row.get(column):
            raise ValueError(f"Row {index + 1} has no {column!r} value")
        groups[row[column]].append(index)
    counts = allocate({name: len(members) for name, members in groups.items()}, total)
    chosen = set()
    for name in sorted(groups):
        # Separate per-category RNG streams keep selections stable if N changes elsewhere.
        rng = random.Random(f"{seed}:{name}")
        chosen.update(rng.sample(groups[name], counts[name]))
    return [rows[index] for index in sorted(chosen)], counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=api.DATA / "test.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True, help="Total subset size N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--column", default="category")
    args = parser.parse_args(argv)
    if args.output.resolve() == args.input.resolve():
        parser.error("--output must differ from --input")
    rows = api.read_csv(args.input)
    api.validate_rows(rows)
    subset, counts = stratified_subset(rows, args.size, args.seed, args.column)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(subset)
    data = buffer.getvalue().encode()
    summary = ", ".join(f"{name}={counts[name]}" for name in sorted(counts))
    if args.output.exists():
        # Concurrent jobs share one subset: never rewrite it, only verify it.
        if args.output.read_bytes() != data:
            raise SystemExit(
                f"{args.output} exists with a different selection; use a new path/OUTPUT_DIR"
            )
        print(f"Stratified subset N={len(subset)} seed={args.seed}: {summary} (existing {args.output})")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(args.output)
    print(f"Stratified subset N={len(subset)} seed={args.seed}: {summary} -> {args.output}")


if __name__ == "__main__":
    main()
