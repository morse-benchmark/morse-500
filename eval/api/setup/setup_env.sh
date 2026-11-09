#!/usr/bin/env bash
set -euo pipefail

SCRIPT_API_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_API_DIR/lib/session.sh"
require_session "$SCRIPT_API_DIR"
if ! command -v uv >/dev/null 2>&1; then
    echo "Missing uv. Install uv using your approved system/package-manager process." >&2
    exit 3
fi
if [[ -x "$PYTHON" ]]; then
    echo "Reusing existing virtual environment: $VENV"
else
    uv venv "$VENV"
fi
uv pip install --python "$PYTHON" \
    -r "$API_DIR/requirements-api.txt"
