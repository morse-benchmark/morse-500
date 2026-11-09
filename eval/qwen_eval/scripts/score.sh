#!/usr/bin/env bash
# Run with the benchmark's Qwen extractor served on localhost:8000.
set -euo pipefail
usage() {
    echo "Usage: bash eval/qwen_eval/scripts/score.sh RESULTS_DIR (local Qwen only; no provider/model arguments)"
}
if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    usage
    echo "Specify the run directory, e.g. eval/results/RUN_NAME; CONCURRENCY defaults to 10."
    exit 0
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
PYTHON="${PYTHON:-$REPO/.qwen_venv/bin/python}"
if (( $# != 1 )); then
    usage >&2
    exit 2
fi
[[ -x "$PYTHON" ]] || {
    echo "Python interpreter missing: $PYTHON; run eval/qwen_eval/setup/setup_env.sh." >&2
    exit 3
}
RESULTS="$1"
RESULTS="$(cd "$RESULTS" && pwd)"

shopt -s nullglob
FILES=("$RESULTS"/pred_sz*_*.csv)
if (( ${#FILES[@]} == 0 )); then
    echo "No prediction CSVs found in $RESULTS" >&2
    exit 1
fi
mkdir -p "$RESULTS/extract"
"$PYTHON" - "$RESULTS" <<'PY'
import csv
import sys
from pathlib import Path

for path in sorted(Path(sys.argv[1]).glob("pred_sz*_*.csv")):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 500:
        raise ValueError(f"{path}: expected 500 rows, found {len(rows)}")
PY
for input in "${FILES[@]}"; do
    name="$(basename "$input")"
    "$PYTHON" eval/qwen_eval/extract_answers.py "$input" \
        --concurrency "${CONCURRENCY:-10}" --allow-failed-predictions \
        --out "$RESULTS/extract/${name/pred_/extract_}"
done
"$PYTHON" eval/qwen_eval/plot_table.py \
    --input-dir "$RESULTS/extract" --output "$RESULTS/benchmark_scores.csv"
