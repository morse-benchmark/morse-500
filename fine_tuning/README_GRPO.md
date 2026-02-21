# GRPO fine-tuning (verl) for temporal_reasoning

This folder contains a minimal GRPO pipeline for Qwen3-VL-4B using verl.

## Files
- `grpo_prep.py`: builds JSONL train/val files from temporal_reasoning videos.
- `grpo_reward.py`: simple exact-match reward; customize as needed.
- `verl_grpo_config.yaml`: minimal GRPO config stub for verl.
- `run_grpo.sh`: runs data prep then launches training.

## Quick start
```bash
python /fs/nexus-scratch/mrislam/morse-500/fine_tuning/grpo_prep.py
python -m verl.trainer.main \
  --config-path /fs/nexus-scratch/mrislam/morse-500/fine_tuning \
  --config-name verl_grpo_config
```

## Notes
- The config keys may vary by verl version. Keep `prompt_field`, `answer_field`,
  and the reward function aligned with your local verl schema.
- The JSONL records include both `messages` and plain `question`/`answer` fields
  so you can switch formats without regenerating data.
