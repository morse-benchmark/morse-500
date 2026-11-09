#!/usr/bin/env bash

OPENAI_MODEL_MODELS=("gpt-6-astra" "gpt-6.1-sol" "gpt-6-luna")
ANTHROPIC_MODEL_MODELS=("claude-fable-5-1" "claude-opus-5-5")
OPENAI_AGENT_MODELS=("gpt-5.3-codex" "gpt-6-astra" "gpt-6.1-sol" "gpt-6.1-terra" "gpt-6-luna")
ANTHROPIC_AGENT_MODELS=("claude-sonnet-5-5" "claude-opus-5-5" "claude-fable-5-1")

SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_CPUS="${SLURM_CPUS:-4}"
SLURM_MEMORY="${SLURM_MEMORY:-32G}"
SLURM_TIME="${SLURM_TIME:-3-00:00:00}"
