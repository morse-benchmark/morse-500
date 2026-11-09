# Session guard shared by every eval/api entry point. Paths come only from
# setup/session_env.sh; there are deliberately no fallback defaults.

MORSE_SESSION_DIRS=(API_DIR REPO VIDEO_ROOT RESULTS_ROOT LOG_DIR FRAME_CACHE_ROOT AGENT_BIN)
MORSE_SESSION_FILES=(DATA_CSV)
MORSE_SESSION_PATHS=(VENV PYTHON CODEX_BINARY CLAUDE_BINARY CODEX_CODE_MODE_HOST)

# Prints every problem and returns nonzero; never exits (safe in a sourced shell).
morse_check_session() {
    local expected_api="${1:-}" name status=0
    for name in "${MORSE_SESSION_DIRS[@]}" "${MORSE_SESSION_FILES[@]}" "${MORSE_SESSION_PATHS[@]}"; do
        if [[ -z "${!name:-}" ]]; then
            echo "  $name is not set" >&2
            status=1
        elif [[ "${!name}" != /* ]]; then
            echo "  $name must be an absolute path: ${!name}" >&2
            status=1
        fi
    done
    for name in "${MORSE_SESSION_DIRS[@]}"; do
        if [[ -n "${!name:-}" && ! -d "${!name}" ]]; then
            echo "  $name directory does not exist: ${!name}" >&2
            status=1
        fi
    done
    for name in "${MORSE_SESSION_FILES[@]}"; do
        if [[ -n "${!name:-}" && ! -f "${!name}" ]]; then
            echo "  $name file does not exist: ${!name}" >&2
            status=1
        fi
    done
    if [[ -n "$expected_api" && -d "${API_DIR:-}" ]] \
        && [[ "$(cd -- "$API_DIR" && pwd -P)" != "$(cd -- "$expected_api" && pwd -P)" ]]; then
        echo "  API_DIR ($API_DIR) is not this checkout ($expected_api)" >&2
        status=1
    fi
    return "$status"
}

require_session() {
    local expected_api="${1:-}"
    if [[ "${MORSE_SESSION_ENV:-}" != 1 ]]; then
        echo "SESSION NOT CONFIGURED: review eval/api/setup/session_env.sh, then run" >&2
        echo "  source eval/api/setup/session_env.sh" >&2
        exit 3
    fi
    if ! morse_check_session "$expected_api"; then
        echo "SESSION INVALID: fix eval/api/setup/session_env.sh and source it again" >&2
        exit 3
    fi
}
