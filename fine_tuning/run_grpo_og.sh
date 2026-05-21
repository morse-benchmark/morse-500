#!/usr/bin/env bash
set -euo pipefail

# 1) Build JSONL data from the temporal_reasoning set.
python /fs/nexus-scratch/mrislam/morse-500/fine_tuning/grpo_prep.py

# 2) Launch verl GRPO training.
# NOTE: Adjust this command to your local verl entry point and config format.
python -m verl.trainer.main_ppo \
  --config-path /fs/nexus-scratch/mrislam/morse-500/fine_tuning \
  --config-name verl_grpo_config
