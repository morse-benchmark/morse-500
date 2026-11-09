#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    printf '%s\n' \
        "Usage: bash eval/qwen_eval/setup/setup_env.sh" \
        "Install Qwen requirements into a separate Python 3.12 VENV (default: .qwen_venv)."
    exit 0
fi
if (( $# )); then
    echo "Usage: bash eval/qwen_eval/setup/setup_env.sh" >&2
    exit 2
fi
QWEN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd -- "$QWEN_DIR/../.." && pwd)"
if ! command -v uv >/dev/null 2>&1; then
    echo "Missing uv. Install uv using your approved system/package-manager process." >&2
    exit 3
fi
VENV="${VENV:-$REPO/.qwen_venv}"
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -r "$QWEN_DIR/requirements.txt"
