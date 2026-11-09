#!/usr/bin/env bash
#SBATCH --job-name=morse-qwen-download
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/morse-qwen-download-%j.log

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
QWEN_DIR="$PWD/eval/qwen_eval"
export PYTHON="${PYTHON:-$PWD/.qwen_venv/bin/python}"
export HF_HOME="${HF_HOME:-/fs/nexus-scratch/dgjersta/.cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
bash "$QWEN_DIR/setup/validate.sh" --stage setup
mkdir -p "$HF_HOME"
"$(dirname "$PYTHON")/hf" download Qwen/Qwen2.5-72B-Instruct-AWQ
