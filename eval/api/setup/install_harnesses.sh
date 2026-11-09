#!/usr/bin/env bash
set -euo pipefail

SCRIPT_API_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_API_DIR/lib/session.sh"
require_session "$SCRIPT_API_DIR"
source "$API_DIR/setup/harnesses.lock"
DEST="$AGENT_BIN"
CODEX_SOURCE="${CODEX_SOURCE:-}"
CLAUDE_SOURCE="${CLAUDE_SOURCE:-}"
CODE_MODE_HOST_SOURCE="${CODE_MODE_HOST_SOURCE:-}"
for pair in "CODEX_SOURCE:$CODEX_SOURCE" "CLAUDE_SOURCE:$CLAUDE_SOURCE" \
            "CODE_MODE_HOST_SOURCE:$CODE_MODE_HOST_SOURCE"; do
    name="${pair%%:*}"; path="${pair#*:}"
    [[ -n "$path" && -x "$path" ]] || {
        echo "Set $name to the official native executable source path" >&2
        exit 2
    }
done
verify_elf() {
    local path="$1"
    [[ "$(head -c 4 "$path")" == $'\x7fELF' ]] || {
        echo "Expected a native Linux ELF binary: $path" >&2; exit 2;
    }
}
verify_version() {
    local path="$1" expected="$2" label="$3" actual
    actual="$("$path" --version 2>&1 || true)"
    [[ "$actual" == *"$expected"* ]] || {
        echo "$label version check failed (expected $expected): $actual" >&2; exit 2;
    }
}
mkdir -p "$DEST"
verify_elf "$CODEX_SOURCE"; verify_elf "$CLAUDE_SOURCE"; verify_elf "$CODE_MODE_HOST_SOURCE"
verify_version "$CODEX_SOURCE" "$CODEX_VERSION" Codex
verify_version "$CLAUDE_SOURCE" "$CLAUDE_VERSION" "Claude Code"
install -m 755 "$CODEX_SOURCE" "$DEST/codex-$CODEX_VERSION"
install -m 755 "$CLAUDE_SOURCE" "$DEST/claude-$CLAUDE_VERSION"
install -m 755 "$CODE_MODE_HOST_SOURCE" "$DEST/codex-code-mode-host-$CODEX_CODE_MODE_HOST_VERSION"
ln -sfn "codex-$CODEX_VERSION" "$DEST/codex"
ln -sfn "claude-$CLAUDE_VERSION" "$DEST/claude"
ln -sfn "codex-code-mode-host-$CODEX_CODE_MODE_HOST_VERSION" "$DEST/codex-code-mode-host"
(
    cd "$DEST"
    sha256sum "codex-$CODEX_VERSION" "claude-$CLAUDE_VERSION" \
        "codex-code-mode-host-$CODEX_CODE_MODE_HOST_VERSION" > harnesses.sha256
)
echo "Staged pinned harness binaries in $DEST; checksums recorded in harnesses.sha256"
