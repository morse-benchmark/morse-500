# Qwen3-VL-4B Fine-Tuning (VERL)

This fine-tuning path now uses **VERL GRPO** on `data/train/metadata.jsonl`.

## Quick start
```bash
bash fine_tuning/run_qwen3_vl_sft.sh
```

## What it does
1. Installs required dependencies (including `verl`).
2. Builds train/val JSONL with `fine_tuning/grpo_prep.py`.
3. Launches VERL trainer using `fine_tuning/verl_grpo_config.yaml`.

## Data requirements
Each metadata row in `data/train/metadata.jsonl` should include:
- `file_name`
- `question_text`
- `solution`

## Config file
Main VERL settings are in:
- `fine_tuning/verl_grpo_config.yaml`

Tune this config for batch size, lr, steps, and model path.
