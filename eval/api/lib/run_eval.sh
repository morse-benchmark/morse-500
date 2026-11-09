#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/session.sh"
require_session "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$API_DIR/scripts/config.sh"

usage() {
    echo "Usage: $0 model|agent openai|anthropic [MODEL ...]" >&2
    exit 2
}
(( $# >= 2 )) || usage
MODE="$1"
PROVIDER="$2"
shift 2
[[ "$MODE" == model || "$MODE" == agent ]] || usage
[[ "$PROVIDER" == openai || "$PROVIDER" == anthropic ]] || usage

case "$MODE/$PROVIDER" in
    model/openai) MODELS=("${OPENAI_MODEL_MODELS[@]}"); KEY=OPENAI_API_KEY ;;
    model/anthropic) MODELS=("${ANTHROPIC_MODEL_MODELS[@]}"); KEY=ANTHROPIC_API_KEY ;;
    agent/openai) MODELS=("${OPENAI_AGENT_MODELS[@]}"); KEY=OPENAI_API_KEY; HARNESS=codex ;;
    agent/anthropic) MODELS=("${ANTHROPIC_AGENT_MODELS[@]}"); KEY=ANTHROPIC_API_KEY; HARNESS=claude ;;
esac
(( $# == 0 )) || MODELS=("$@")
(( ${#MODELS[@]} > 0 )) || { echo "No models configured for $MODE/$PROVIDER" >&2; exit 2; }
for model in "${MODELS[@]}"; do
    [[ "$model" =~ ^[A-Za-z0-9._-]+$ ]] || {
        echo "Invalid model ID: $model" >&2
        exit 2
    }
done

INPUT="$DATA_CSV"
FPS="${FPS:-4}"
MAX_FRAMES="${MAX_FRAMES:-32}"
MAX_SIDE="${MAX_SIDE:-512}"
LIMIT="${LIMIT:-}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-$([[ "$MODE" == agent ]] && echo 8192 || echo 4096)}"
RPM="${RPM:-5}"
ACTION="${ACTION:-run}"

fail() { echo "PREFLIGHT FAILED: $*" >&2; exit 3; }
[[ -x "$PYTHON" ]] || fail "Python interpreter not executable: $PYTHON (run setup/setup_env.sh)"
[[ -f "$INPUT" ]] || fail "Benchmark CSV not found: $INPUT"
[[ -d "$VIDEO_ROOT" ]] || fail "Video root not found: $VIDEO_ROOT"
[[ -n "${!KEY:-}" ]] || fail "Export $KEY before running; do not put credentials in scripts"
"$PYTHON" -c 'import anthropic, moviepy, numpy, openai, pandas, PIL' \
    || fail "API Python dependencies missing (run setup/setup_env.sh)"

RUN_ARGS=()
if [[ "${PREFLIGHT_ONLY:-0}" != 1 ]]; then
    [[ -z "${RUN_NAME:-}" ]] || RUN_ARGS+=(--run-name "$RUN_NAME")
    [[ -z "${RESUME_RUN:-}" ]] || RUN_ARGS+=(--resume-run "$RESUME_RUN")
    [[ -z "${OUTPUT_DIR:-}" ]] || RUN_ARGS+=(--output-dir "$OUTPUT_DIR")
    [[ -z "${BUDGET_FILE:-}" ]] || RUN_ARGS+=(--budget-file "$BUDGET_FILE")
    [[ -z "${ESTIMATE_DIR:-}" ]] || RUN_ARGS+=(--estimate-dir "$ESTIMATE_DIR")
    OUTPUT_DIR="$("$PYTHON" "$API_DIR/lib/run_paths.py" "${RUN_ARGS[@]}" \
        --results-root "$RESULTS_ROOT" --prefix "$MODE-$PROVIDER-$ACTION")" \
        || fail "Run initialization failed"
    RUN_DIR="$OUTPUT_DIR"
    RUN_ARGS=(--resume-run "${RUN_DIR##*/}")
    echo "Run directory: $RUN_DIR"
fi
BUDGET_FILE="${BUDGET_FILE:-${OUTPUT_DIR:-$RESULTS_ROOT}/run_usage/budget.jsonl}"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SAMPLING=(--fps "$FPS" --max-frames "$MAX_FRAMES" --max-side "$MAX_SIDE")
DATA_ARGS=(--input "$INPUT" --video-root "$VIDEO_ROOT")
LIMIT_ARGS=()
[[ -z "$LIMIT" ]] || LIMIT_ARGS+=(--limit "$LIMIT")
MODEL_EXTRA=()
[[ -z "${REASONING_EFFORT:-}" || "$PROVIDER" != openai ]] || MODEL_EXTRA+=(--reasoning-effort "$REASONING_EFFORT")
[[ "$PROVIDER" != openai ]] || MODEL_EXTRA+=(--openai-endpoint "${OPENAI_ENDPOINT:-chat}")
FRAME_CACHE="${FRAME_CACHE:-$FRAME_CACHE_ROOT/fps${FPS}_max${MAX_FRAMES}_sz${MAX_SIDE}}"
PREPROCESS_MODE="${PREPROCESS_MODE:-cached}"
BUDGET_ARGS=()
if [[ -n "${BUDGET_USD:-}" ]]; then
    BUDGET_ARGS=(--budget-usd "$BUDGET_USD" --budget-file "$BUDGET_FILE")
fi

preflight() {
    if [[ "$MODE" == agent ]]; then
        command -v bwrap >/dev/null || fail "Missing agent isolation dependency: bwrap"
        command -v "${SOCAT_BINARY:-socat}" >/dev/null \
            || fail "Missing agent relay dependency: ${SOCAT_BINARY:-socat}"
        [[ "$(uname -s)" == Linux ]] || fail "Agent harness isolation requires Linux/bwrap"
        [[ -x "$CODEX_BINARY" || "$HARNESS" != codex ]] \
            || fail "Codex binary missing; run setup/install_harnesses.sh"
        [[ -x "$CLAUDE_BINARY" || "$HARNESS" != claude ]] \
            || fail "Claude binary missing; run setup/install_harnesses.sh"
    fi
    for model in "${MODELS[@]}"; do
        if [[ -n "${BUDGET_USD:-}" || "${ESTIMATE:-none}" != none
                || "${BUDGET_SOURCE:-fixed}" != fixed ]]; then
            "$PYTHON" - "$API_DIR/estimates/prices.json" "$PROVIDER/$model" <<'PY' \
                || fail "Missing price entry for $PROVIDER/$model"
import json, sys
sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1]))["models"] else 1)
PY
        fi
        if [[ "$MODE" == agent ]]; then
            binary="$CODEX_BINARY"
            [[ "$HARNESS" != claude ]] || binary="$CLAUDE_BINARY"
            "$PYTHON" "$API_DIR/agents/run_agents_eval.py" --harness "$HARNESS" \
                --model "$model" --binary "$binary" --preflight-only --check-model \
                --max-output-tokens "$MAX_OUTPUT_TOKENS" \
                "${BUDGET_ARGS[@]}" || fail "Agent preflight failed for $model"
        else
            "$PYTHON" "$API_DIR/utils/check_api.py" "$PROVIDER" --model "$model" \
                || fail "Model preflight failed for $PROVIDER/$model"
        fi
    done
}

run_models() {
    mkdir -p "$OUTPUT_DIR" || return $?
    local frame_args=()
    case "$PREPROCESS_MODE" in
        cached)
            frame_args=(--frame-cache "$FRAME_CACHE")
            "$PYTHON" "$API_DIR/lib/frames.py" prepare "${DATA_ARGS[@]}" "${SAMPLING[@]}" \
                "${frame_args[@]}" "${LIMIT_ARGS[@]}" || return $?
            ;;
        on-demand) echo "On-demand frame preprocessing selected" ;;
        *) fail "PREPROCESS_MODE must be cached or on-demand" ;;
    esac
    local failed=0 status output
    local count_args=()
    if [[ -z "$LIMIT" ]]; then count_args=(--expected-count 500); fi
    for model in "${MODELS[@]}"; do
        output="$OUTPUT_DIR/pred_sz${MAX_SIDE}_${PROVIDER}_${model}_fps${FPS}_max${MAX_FRAMES}.csv"
        status=0
        "$PYTHON" "$API_DIR/models/run_models_eval.py" predict --provider "$PROVIDER" \
            --model "$model" "${DATA_ARGS[@]}" "${SAMPLING[@]}" "${frame_args[@]}" \
            --concurrency "${CONCURRENCY:-${SLURM_CPUS_PER_TASK:-2}}" --rpm "$RPM" \
            --max-output-tokens "$MAX_OUTPUT_TOKENS" --timeout "${TIMEOUT:-300}" \
            --attempts "${ATTEMPTS:-5}" --output "$output" "${LIMIT_ARGS[@]}" \
            "${count_args[@]}" "${MODEL_EXTRA[@]}" \
            "${BUDGET_ARGS[@]}" "${RUN_ARGS[@]}" || status=$?
        if (( status == 76 )); then return 76; fi
        (( status == 0 )) || failed=1
    done
    return "$failed"
}

run_agents() {
    mkdir -p "$OUTPUT_DIR" || return $?
    "$PYTHON" "$API_DIR/lib/frames.py" prepare "${DATA_ARGS[@]}" "${SAMPLING[@]}" \
        --frame-cache "$FRAME_CACHE" "${LIMIT_ARGS[@]}" || return $?
    local failed=0 status model binary
    binary="$CODEX_BINARY"
    [[ "$HARNESS" != claude ]] || binary="$CLAUDE_BINARY"
    for model in "${MODELS[@]}"; do
        status=0
        "$PYTHON" "$API_DIR/agents/run_agents_eval.py" --harness "$HARNESS" --model "$model" \
            --binary "$binary" "${DATA_ARGS[@]}" "${SAMPLING[@]}" "${LIMIT_ARGS[@]}" \
            --frame-cache "$FRAME_CACHE" --max-output-tokens "$MAX_OUTPUT_TOKENS" \
            --rpm "$RPM" --max-requests "${MAX_REQUESTS:-30}" \
            --seconds "${SECONDS_PER_EXAMPLE:-1800}" \
            --max-consecutive-failures "${MAX_CONSECUTIVE_FAILURES:-3}" \
            --output "$OUTPUT_DIR/pred_sz${MAX_SIDE}_${HARNESS}_${model}_fps${FPS}_max${MAX_FRAMES}.csv" \
            "${BUDGET_ARGS[@]}" "${RUN_ARGS[@]}" || status=$?
        if (( status == 76 )); then return 76; fi
        if (( status == 75 )); then return 75; fi
        (( status == 0 )) || failed=1
    done
    return "$failed"
}

make_pilot() {
    local full_input="$INPUT" full_output="$OUTPUT_DIR" subset_size="${SUBSET_SIZE:-60}"
    local pilot_dir="${ESTIMATE_DIR:-$full_output/estimate}"
    mkdir -p "$pilot_dir" || return $?
    "$PYTHON" "$API_DIR/estimates/make_subset.py" --input "$full_input" \
        --output "$pilot_dir/subset.csv" --size "$subset_size" --seed "${SUBSET_SEED:-0}" \
        || return $?
    INPUT="$pilot_dir/subset.csv"
    OUTPUT_DIR="$pilot_dir"
    LIMIT="$subset_size"
    DATA_ARGS=(--input "$INPUT" --video-root "$VIDEO_ROOT")
    LIMIT_ARGS=(--limit "$subset_size")
    local status=0
    if [[ "$MODE" == agent ]]; then
        run_agents || status=$?
    else
        run_models || status=$?
    fi
    "$PYTHON" "$API_DIR/estimates/estimate_costs.py" "$OUTPUT_DIR" \
        --subset "$pilot_dir/subset.csv" --full "$full_input" --target "${TARGET:-500}" \
        --output "$pilot_dir/estimate.json" --markdown "$pilot_dir/estimate.md" \
        || return $?
    INPUT="$full_input"
    OUTPUT_DIR="$full_output"
    LIMIT="${ORIGINAL_LIMIT:-}"
    DATA_ARGS=(--input "$INPUT" --video-root "$VIDEO_ROOT")
    LIMIT_ARGS=()
    [[ -z "$LIMIT" ]] || LIMIT_ARGS+=(--limit "$LIMIT")
    return "$status"
}

ORIGINAL_LIMIT="$LIMIT"
if [[ "$ACTION" == estimate ]]; then
    [[ "${METHOD:-pilot}" == pilot ]] || fail "Only METHOD=pilot is supported"
    preflight
    if [[ "${PREFLIGHT_ONLY:-0}" != 1 ]]; then make_pilot; fi
    exit 0
elif [[ "$ACTION" != run ]]; then
    fail "ACTION must be run or estimate"
fi

preflight
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
    echo "Preflight complete for $MODE/$PROVIDER (${MODELS[*]})"
    exit 0
fi
if [[ "${ESTIMATE:-none}" == bootstrap ]]; then
    make_pilot
    ESTIMATE="file:${ESTIMATE_DIR:-$OUTPUT_DIR/estimate}/estimate.json"
elif [[ "${ESTIMATE:-none}" != none && "${ESTIMATE:-none}" != file:* ]]; then
    fail "ESTIMATE must be none, bootstrap, or file:/path/to/estimate.json"
fi

BUDGET_SOURCE="${BUDGET_SOURCE:-fixed}"
BUDGET_POLICY="${BUDGET_POLICY:-$([[ -n "${BUDGET_USD:-}" ]] && echo cap || echo off)}"
if [[ "$BUDGET_POLICY" != off ]]; then
    [[ "$BUDGET_POLICY" == gate || "$BUDGET_POLICY" == cap || "$BUDGET_POLICY" == both ]] \
        || fail "BUDGET_POLICY must be off, gate, cap, or both"
fi
case "$BUDGET_SOURCE" in
    fixed)
        [[ "$BUDGET_POLICY" != gate && "$BUDGET_POLICY" != both ]] \
            || fail "gate policy requires BUDGET_SOURCE=estimate-point or estimate-upper"
        if [[ "$BUDGET_POLICY" == cap || "$BUDGET_POLICY" == both ]]; then
            [[ -n "${BUDGET_USD:-}" ]] || fail "BUDGET_USD is required for fixed budget source"
        fi
        ;;
    estimate-point|estimate-upper)
        [[ "${ESTIMATE:-}" == file:* ]] || fail "BUDGET_SOURCE requires ESTIMATE=bootstrap or ESTIMATE=file:PATH"
        estimate_path="${ESTIMATE#file:}"
        estimate_value="$("$PYTHON" "$API_DIR/estimates/read_estimate.py" \
            "$estimate_path" "$PROVIDER" "${MODELS[@]}" \
            --kind "${BUDGET_SOURCE#estimate-}")" \
            || fail "Could not read requested models from estimate: $estimate_path"
        if [[ "$BUDGET_POLICY" == gate || "$BUDGET_POLICY" == both ]]; then
            [[ -n "${BUDGET_USD:-}" ]] || fail "BUDGET_USD is required for estimate gate policy"
            "$PYTHON" - "$estimate_value" "$BUDGET_USD" <<'PY' \
                || { echo "BUDGET GATE: estimate exceeds BUDGET_USD" >&2; exit 77; }
import sys
sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)
PY
        fi
        if [[ "$BUDGET_POLICY" == cap || "$BUDGET_POLICY" == both ]]; then
            BUDGET_USD="$("$PYTHON" - "$estimate_value" "${BUDGET_MARGIN:-1.2}" \
                "${BUDGET_USD:-inf}" "$BUDGET_POLICY" <<'PY'
import math, sys
estimate, margin, fixed = map(float, sys.argv[1:4])
if not math.isfinite(estimate) or estimate < 0 or not math.isfinite(margin) or margin <= 0:
    raise SystemExit("Estimate and BUDGET_MARGIN must be finite; margin must be positive")
cap = estimate * margin
if sys.argv[4] == "both":
    cap = min(cap, fixed)
print(f"{cap:.8f}")
PY
            )" || fail "Could not derive an estimate-based budget cap"
            BUDGET_FILE="${BUDGET_FILE:-$RUN_DIR/run_usage/budget.jsonl}"
            BUDGET_ARGS=(--budget-usd "$BUDGET_USD" --budget-file "$BUDGET_FILE")
            echo "Estimate-based cap: \$${BUDGET_USD} (margin ${BUDGET_MARGIN:-1.2})"
        fi
        ;;
    *) fail "BUDGET_SOURCE must be fixed, estimate-point, or estimate-upper" ;;
esac
if [[ "$BUDGET_POLICY" == gate ]]; then BUDGET_ARGS=(); fi

status=0
if [[ "$MODE" == model ]]; then
    run_models || status=$?
else
    run_agents || status=$?
fi
exit "$status"
