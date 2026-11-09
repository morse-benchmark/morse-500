#!/usr/bin/env bash
#SBATCH --job-name=morse-qwen-score
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/morse-qwen-score-%j.log

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
QWEN_DIR="$PWD/eval/qwen_eval"
export HF_HOME="${HF_HOME:-/fs/nexus-scratch/dgjersta/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHON="${PYTHON:-$PWD/.qwen_venv/bin/python}"
export CUDA_HOME="${CUDA_HOME:-/opt/common/cuda/cuda-13.0.2}"
export PATH="$(dirname "$PYTHON"):$CUDA_HOME/bin:$PATH"
export OPENAI_API_KEY=EMPTY
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1"
unset OPENAI_BASE_URL OPENAI_ORG_ID OPENAI_PROJECT_ID ANTHROPIC_API_KEY

bash "$QWEN_DIR/setup/validate.sh" --stage gpu
"$PYTHON" - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 8000))
PY
"$PYTHON" -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-72B-Instruct-AWQ \
    --served-model-name Qwen/Qwen2.5-72B-Instruct-AWQ \
    --host 127.0.0.1 --port 8000 \
    --dtype half --max-model-len 32768 \
    --max-num-seqs 10 --gpu-memory-utilization 0.85 \
    --generation-config vllm &
SERVER_PID=$!
cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ready=0
for (( attempt=0; attempt<180; attempt++ )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Qwen server exited during startup; see this job's log." >&2
        exit 1
    fi
    if curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 10
done
if (( ! ready )); then
    echo "Qwen server did not become ready within 30 minutes." >&2
    exit 1
fi
bash "$QWEN_DIR/setup/validate.sh" --stage server
bash "$QWEN_DIR/scripts/score.sh" "$@"
