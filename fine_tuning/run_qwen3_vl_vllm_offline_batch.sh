#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${1:-Qwen/Qwen3-VL-4B-Instruct}"

if [[ ! -d "${ROOT_DIR}/.morse_venv" ]]; then
  python3 -m venv "${ROOT_DIR}/.morse_venv"
fi

source "${ROOT_DIR}/.morse_venv/bin/activate"
python -m pip install --upgrade pip
pip install -U vllm transformers

python "${ROOT_DIR}/fine_tuning/infer_qwen3_vl_vllm_offline_batch.py" \
  --model "${MODEL_NAME}" \
  --metadata "${ROOT_DIR}/data/train/metadata.jsonl" \
  --dataset-root "${ROOT_DIR}/data/train" \
  --output "${ROOT_DIR}/fine_tuning/checkpoints/vllm_batch_predictions.jsonl" \
  --batch-size 4 \
  --max-tokens 128
