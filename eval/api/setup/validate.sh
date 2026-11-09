#!/usr/bin/env bash
set -euo pipefail

SCRIPT_API_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_API_DIR/lib/session.sh"
require_session "$SCRIPT_API_DIR"
MODE=""
PROVIDER=""
while (($#)); do
    case "$1" in
        --mode) MODE="${2:?--mode requires model or agent}"; shift 2 ;;
        --provider) PROVIDER="${2:?--provider requires openai or anthropic}"; shift 2 ;;
        *) echo "Usage: $0 --mode model|agent --provider openai|anthropic" >&2; exit 2 ;;
    esac
done
[[ "$MODE" == model || "$MODE" == agent ]] || { echo "Invalid --mode" >&2; exit 2; }
[[ "$PROVIDER" == openai || "$PROVIDER" == anthropic ]] || { echo "Invalid --provider" >&2; exit 2; }
INPUT="$DATA_CSV"
[[ -x "$PYTHON" ]] || { echo "Python interpreter missing: $PYTHON; run setup/setup_env.sh" >&2; exit 3; }
[[ -f "$INPUT" ]] || { echo "Dataset CSV missing: $INPUT" >&2; exit 3; }
[[ -d "$VIDEO_ROOT" ]] || { echo "Video directory missing: $VIDEO_ROOT" >&2; exit 3; }
"$PYTHON" -c 'import anthropic, moviepy, numpy, openai, pandas, PIL' || {
    echo "Required Python dependencies are missing; run setup/setup_env.sh" >&2
    exit 3
}
KEY=OPENAI_API_KEY
[[ "$PROVIDER" != anthropic ]] || KEY=ANTHROPIC_API_KEY
[[ -n "${!KEY:-}" ]] || { echo "Export $KEY before evaluation" >&2; exit 3; }
if [[ "$MODE" == agent ]]; then
    [[ "$(uname -s)" == Linux ]] || { echo "Agent harnesses require Linux namespaces" >&2; exit 3; }
    command -v bwrap >/dev/null || { echo "Agent dependency missing: bwrap" >&2; exit 3; }
    command -v "${SOCAT_BINARY:-socat}" >/dev/null || {
        echo "Agent relay dependency missing: ${SOCAT_BINARY:-socat}" >&2; exit 3;
    }
    namespace_probe=(bwrap --unshare-all --die-with-parent)
    for path in /usr /bin /lib /lib64; do
        [[ ! -e "$path" ]] || namespace_probe+=(--ro-bind "$path" "$path")
    done
    namespace_probe+=(--proc /proc --dev /dev /usr/bin/true)
    if ! "${namespace_probe[@]}" >/dev/null 2>&1; then
        echo "Bubblewrap/user namespaces are not usable on this host" >&2
        exit 3
    fi
    if [[ "$PROVIDER" == openai ]]; then
        BINARY="$CODEX_BINARY"
        HOST="$CODEX_CODE_MODE_HOST"
    else
        BINARY="$CLAUDE_BINARY"
        HOST=""
    fi
    [[ -x "$BINARY" ]] || { echo "Harness binary missing: $BINARY; stage with setup/install_harnesses.sh" >&2; exit 3; }
    [[ -z "$HOST" || -x "$HOST" ]] || { echo "Codex code-mode host missing: $HOST" >&2; exit 3; }
    "$PYTHON" - "$BINARY" "$HOST" <<'PY'
from pathlib import Path
import os, sys
for name in sys.argv[1:]:
    if not name:
        continue
    path = Path(name)
    if path.read_bytes()[:4] != b"\x7fELF" or not os.access(path, os.X_OK):
        raise SystemExit(f"Not an executable Linux ELF: {path}")
PY
fi
echo "PASS: $MODE/$PROVIDER prerequisites are present on $(hostname)"
