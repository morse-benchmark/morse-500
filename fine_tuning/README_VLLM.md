# vLLM Offline Batch Inference

This workflow uses **vLLM offline batch mode only** (no server mode).

## Run
```bash
bash fine_tuning/run_qwen3_vl_vllm_offline_batch.sh
```

## Manual run
```bash
python fine_tuning/infer_qwen3_vl_vllm_offline_batch.py \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --metadata data/train/metadata.jsonl \
  --dataset-root data/train \
  --output fine_tuning/checkpoints/vllm_batch_predictions.jsonl \
  --batch-size 4 \
  --max-tokens 128
```

## Notes
- Input records are read from `data/train/metadata.jsonl`.
- Requests are sent to vLLM via `LLM.generate(...)` in batches (`--batch-size`).
- Output is JSONL with `question`, `target`, and `prediction`.
