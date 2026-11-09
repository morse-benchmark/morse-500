#!/usr/bin/env bash
set -euo pipefail

SCRIPT_API_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SCRIPT_API_DIR/lib/session.sh"
require_session "$SCRIPT_API_DIR"
source "$API_DIR/scripts/config.sh"
if ! command -v sbatch >/dev/null 2>&1; then
    echo "Required command is missing: sbatch (run this entry point on a Slurm submit host)" >&2
    exit 3
fi
args=(--chdir "$REPO" --output "$LOG_DIR/morse-api-check-%j.log" --export "ALL")
[[ -z "$SLURM_PARTITION" ]] || args+=(--partition "$SLURM_PARTITION")
[[ -z "$SLURM_ACCOUNT" ]] || args+=(--account "$SLURM_ACCOUNT")
[[ -z "$SLURM_QOS" ]] || args+=(--qos "$SLURM_QOS")
exec sbatch "${args[@]}" "$API_DIR/scripts/extras/check_api.sbatch" "$@"
