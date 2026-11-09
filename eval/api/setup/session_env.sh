# MORSE-500 API evaluation session setup.
#
# 1. Fill in every value in the review block below with an absolute path.
#    Nothing is pre-filled and nothing falls back to a default: sourcing fails
#    until every value is set and points to an existing location.
# 2. From the repository root, load it into your current shell:
#        source eval/api/setup/session_env.sh
#    It must be sourced, not run with `bash`, so the variables and keys persist
#    in this shell and propagate to Slurm jobs submitted from it.
#
# All eval/api scripts refuse to run until this file has been sourced
# successfully; they have no fallback paths of their own.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this file instead of executing it: source ${BASH_SOURCE[0]}" >&2
    exit 2
fi

unset MORSE_SESSION_ENV

# ---------------------------------------------------------------- review ---
# Repository checkout that contains eval/api (e.g. /path/to/morse-500).
export REPO=""
# Benchmark inputs; must already exist
# (e.g. /path/to/datasets/morse-500/test.csv and /path/to/datasets/morse-500/test_sz512).
export DATA_CSV=""
export VIDEO_ROOT=""
# API Python virtual environment; setup/setup_env.sh creates it
# (e.g. /path/to/morse-500/.morse_venv). Its parent directory must exist.
export VENV=""
# Output directories; each is created if missing, but its parent must exist
# (e.g. /path/to/morse-500/eval/results, /path/to/morse-500/logs,
# /path/to/morse-500/eval/api/frame_cache).
export RESULTS_ROOT=""
export LOG_DIR=""
export FRAME_CACHE_ROOT=""
# Directory holding the agent harness binaries; setup/install_harnesses.sh
# stages into it (e.g. /path/to/agent-bin). Created if missing; parent must exist.
export AGENT_BIN=""
# ---------------------------------------------------------------------------

# Derived from the values above (empty while their base is empty).
export API_DIR="${REPO:+$REPO/eval/api}"
export PYTHON="${VENV:+$VENV/bin/python}"
export CODEX_BINARY="${AGENT_BIN:+$AGENT_BIN/codex}"
export CLAUDE_BINARY="${AGENT_BIN:+$AGENT_BIN/claude}"
export CODEX_CODE_MODE_HOST="${AGENT_BIN:+$AGENT_BIN/codex-code-mode-host}"

morse_session_unfilled=()
for morse_session_name in REPO DATA_CSV VIDEO_ROOT VENV RESULTS_ROOT LOG_DIR FRAME_CACHE_ROOT AGENT_BIN; do
    if [[ -z "${!morse_session_name}" ]]; then
        morse_session_unfilled+=("  $morse_session_name is not filled in")
    elif [[ "${!morse_session_name}" != /* ]]; then
        morse_session_unfilled+=("  $morse_session_name must be an absolute path: ${!morse_session_name}")
    fi
done
for morse_session_name in VENV RESULTS_ROOT LOG_DIR FRAME_CACHE_ROOT AGENT_BIN; do
    morse_session_value="${!morse_session_name}"
    if [[ "$morse_session_value" == /* && ! -d "$(dirname -- "$morse_session_value")" ]]; then
        morse_session_unfilled+=("  $morse_session_name parent directory does not exist: $(dirname -- "$morse_session_value")")
    fi
done
if (( ${#morse_session_unfilled[@]} )); then
    printf '%s\n' "${morse_session_unfilled[@]}" >&2
    echo "SESSION SETUP FAILED: fill in the review block in ${BASH_SOURCE[0]} and source it again" >&2
    unset morse_session_name morse_session_value morse_session_unfilled
    return 1
fi
unset morse_session_name morse_session_value morse_session_unfilled

# Only the final directory level is created, so a mistyped parent fails here.
if ! mkdir -p -- "$RESULTS_ROOT" "$LOG_DIR" "$FRAME_CACHE_ROOT" "$AGENT_BIN"; then
    echo "SESSION SETUP FAILED: could not create output directories" >&2
    return 1
fi
if ! ( source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/session.sh" \
        && morse_check_session "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" ); then
    echo "SESSION SETUP FAILED: fix the paths in ${BASH_SOURCE[0]} and source it again" >&2
    return 1
fi
[[ -x "$PYTHON" ]] || echo "Note: $PYTHON not found yet; run: bash eval/api/setup/setup_env.sh" >&2

# Keys are read silently: never echoed, stored in history, or passed as arguments.
read -rsp "OpenAI API key: " OPENAI_API_KEY; echo >&2
read -rsp "Anthropic API key: " ANTHROPIC_API_KEY; echo >&2
export OPENAI_API_KEY ANTHROPIC_API_KEY
if [[ -z "$OPENAI_API_KEY" || -z "$ANTHROPIC_API_KEY" ]]; then
    echo "SESSION SETUP FAILED: both OPENAI_API_KEY and ANTHROPIC_API_KEY are required" >&2
    return 1
fi

export MORSE_SESSION_ENV=1
printf 'MORSE session ready.\n  REPO=%s\n  RESULTS_ROOT=%s\n  LOG_DIR=%s\n' "$REPO" "$RESULTS_ROOT" "$LOG_DIR"
