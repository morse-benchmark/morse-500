#!/usr/bin/env bash
#SBATCH --job-name=morse-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/morse-eval-%j.log

set -euo pipefail

API_DIR="${API_DIR:?Source setup/session_env.sh, then submit with scripts/submit_slurm.sh}"
source "$API_DIR/lib/run_eval.sh"
