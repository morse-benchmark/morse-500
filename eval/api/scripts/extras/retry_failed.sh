#!/usr/bin/env bash
set -euo pipefail

SCRIPT_API_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_API_DIR/lib/session.sh"
require_session "$SCRIPT_API_DIR"
cd "$REPO"
PROVIDER="${1:?Usage: bash eval/api/scripts/extras/retry_failed.sh openai|anthropic MODEL}"
MODEL="${2:?Usage: bash eval/api/scripts/extras/retry_failed.sh openai|anthropic MODEL}"
if (( $# != 2 )); then
    echo "Expected exactly one provider and one model" >&2
    exit 2
fi
if [[ ! "$MODEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid model ID: $MODEL" >&2
    exit 2
fi
case "$PROVIDER" in
    openai) : "${OPENAI_API_KEY:?Export OPENAI_API_KEY before submitting}" ;;
    anthropic) : "${ANTHROPIC_API_KEY:?Export ANTHROPIC_API_KEY before submitting}" ;;
    *) echo "Unknown provider: $PROVIDER" >&2; exit 2 ;;
esac

FPS="${FPS:-4}"
MAX_FRAMES="${MAX_FRAMES:-32}"
MAX_SIDE="${MAX_SIDE:-512}"
CONCURRENCY="${CONCURRENCY:-2}"
RPM="${RPM:-5}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-8192}"
TIMEOUT="${TIMEOUT:-300}"
ATTEMPTS="${ATTEMPTS:-5}"
OPENAI_ENDPOINT="${OPENAI_ENDPOINT:-chat}"
SOURCE_CSV="$DATA_CSV"
: "${ORIGINAL_RESULTS:?Specify ORIGINAL_RESULTS=\$RESULTS_ROOT/RUN_NAME}"
RUN_ARGS=()
[[ -z "${RUN_NAME:-}" ]] || RUN_ARGS+=(--run-name "$RUN_NAME")
[[ -z "${RESUME_RUN:-}" ]] || RUN_ARGS+=(--resume-run "$RESUME_RUN")
[[ -z "${OUTPUT_DIR:-}" ]] || RUN_ARGS+=(--output-dir "$OUTPUT_DIR")
RECOVERY_DIR="$("$PYTHON" "$API_DIR/lib/run_paths.py" "${RUN_ARGS[@]}" \
    --results-root "$RESULTS_ROOT" --prefix "model-$PROVIDER-recovery")"
echo "Run directory: $RECOVERY_DIR"
FRAME_CACHE="${FRAME_CACHE:-$FRAME_CACHE_ROOT/fps${FPS}_max${MAX_FRAMES}_sz${MAX_SIDE}}"
INPUT="$RECOVERY_DIR/inputs/failed_sz${MAX_SIDE}_${PROVIDER}_${MODEL}_fps${FPS}_max${MAX_FRAMES}.csv"
OUTPUT="$RECOVERY_DIR/pred_retry_sz${MAX_SIDE}_${PROVIDER}_${MODEL}_fps${FPS}_max${MAX_FRAMES}.csv"
ORIGINAL="$ORIGINAL_RESULTS/pred_sz${MAX_SIDE}_${PROVIDER}_${MODEL}_fps${FPS}_max${MAX_FRAMES}.csv"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$PYTHON" eval/api/utils/prepare_failed_rows.py \
    --input "$SOURCE_CSV" --predictions "$ORIGINAL" --output "$INPUT"
if [[ ! -s "$INPUT" ]]; then
    echo "No failed rows to retry for $PROVIDER/$MODEL" >&2
    exit 0
fi

ARGS=()
if [[ "$PROVIDER" == openai ]]; then
    ARGS+=(--openai-endpoint "$OPENAI_ENDPOINT")
fi

"$PYTHON" eval/api/models/run_models_eval.py predict \
    --provider "$PROVIDER" --model "$MODEL" \
    --input "$INPUT" --video-root "$VIDEO_ROOT" \
    --frame-cache "$FRAME_CACHE" \
    --fps "$FPS" --max-frames "$MAX_FRAMES" --max-side "$MAX_SIDE" \
    --concurrency "$CONCURRENCY" --rpm "$RPM" \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --timeout "$TIMEOUT" --attempts "$ATTEMPTS" \
    --output "$OUTPUT" --resume-run "${RECOVERY_DIR##*/}" "${ARGS[@]}"
