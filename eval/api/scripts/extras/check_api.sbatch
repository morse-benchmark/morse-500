#!/usr/bin/env bash
#SBATCH --job-name=morse-api-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/morse-api-check-%j.log

set -euo pipefail
API_DIR="${API_DIR:?Source setup/session_env.sh, then submit with scripts/extras/submit_check_slurm.sh}"
source "$API_DIR/lib/session.sh"
require_session "$API_DIR"
if (($# == 0)); then set -- both; fi
exec "$PYTHON" "$API_DIR/utils/check_api.py" "$@"
