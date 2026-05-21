#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
ROLLOUT_N="${ROLLOUT_N:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:--1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-vl-4b-verl-grpo}"

if [[ ! -d "${ROOT_DIR}/.morse_venv" ]]; then
  python3 -m venv "${ROOT_DIR}/.morse_venv"
fi
source "${ROOT_DIR}/.morse_venv/bin/activate"

python -m pip install --upgrade pip
pip install -U \
  "torch>=2.2" torchvision torchaudio \
  "transformers==5.2.0" datasets pandas pyarrow \
  peft accelerate bitsandbytes qwen-vl-utils av decord ray \
  verl

python - <<'PY'
from __future__ import annotations

from pathlib import Path
import verl

PATCH_TAG = "VERL_QWEN3_COMPAT_PATCH"
base = Path(verl.__file__).resolve().parent
files = [
    base / "utils" / "model.py",
    base / "model_merger" / "base_model_merger.py",
    base / "workers" / "fsdp_workers.py",
    base / "models" / "transformers" / "qwen2_vl.py",
    base / "models" / "transformers" / "qwen3_vl.py",
    base / "utils" / "tensordict_utils.py",
]

for path in files:
    text = path.read_text(encoding="utf-8")
    if PATCH_TAG in text:
        continue
    original = text
    text = text.replace("    AutoModelForVision2Seq,\n", "")

    if path.name == "model.py":
        needle = "from transformers.modeling_outputs import CausalLMOutputWithPast\n"
        inject = (
            "try:\n"
            "    from transformers import AutoModelForVision2Seq\n"
            "except ImportError:\n"
            "    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq\n"
            f"# {PATCH_TAG}\n\n"
            "from transformers.modeling_outputs import CausalLMOutputWithPast\n"
        )
        text = text.replace(needle, inject)
    elif path.name == "base_model_merger.py":
        needle = "from verl.utils import hf_processor, hf_tokenizer\n"
        inject = (
            "try:\n"
            "    from transformers import AutoModelForVision2Seq\n"
            "except ImportError:\n"
            "    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq\n"
            f"# {PATCH_TAG}\n\n"
            "from verl.utils import hf_processor, hf_tokenizer\n"
        )
        text = text.replace(needle, inject)
    elif path.name == "fsdp_workers.py":
        needle = "        from verl.utils.model import get_generation_config, print_model_size, update_model_config\n"
        inject = (
            "        try:\n"
            "            from transformers import AutoModelForVision2Seq\n"
            "        except ImportError:\n"
            "            from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq\n"
            f"        # {PATCH_TAG}\n\n"
            "        from verl.utils.model import get_generation_config, print_model_size, update_model_config\n"
        )
        text = text.replace(needle, inject)
    elif path.name == "qwen2_vl.py":
        needle = "        image_nums = (vision_tokens == image_token_id).sum()\n        video_nums = (vision_tokens == video_token_id).sum()\n"
        inject = (
            "        image_nums = (vision_tokens == image_token_id).sum()\n"
            "        video_nums = (vision_tokens == video_token_id).sum()\n"
            "        # Qwen3-VL can emit extra vision markers relative to grid tensors.\n"
            "        # Clamp to available grid entries to avoid index overflow in VERL's Qwen2 helper.\n"
            "        if image_grid_thw is not None:\n"
            "            image_nums = min(int(image_nums), int(image_grid_thw.shape[0]))\n"
            "        if video_grid_thw is not None:\n"
            "            video_nums = min(int(video_nums), int(video_grid_thw.shape[0]))\n"
        )
        text = text.replace(needle, inject)
    elif path.name == "qwen3_vl.py":
        text = text.replace(
            "        image_embeds, deepstack_image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)\n",
            "        _image_out = model.visual(pixel_values, grid_thw=image_grid_thw)\n"
            "        if hasattr(_image_out, 'pooler_output'):\n"
            "            image_embeds = _image_out.pooler_output\n"
            "            deepstack_image_embeds = _image_out.deepstack_features\n"
            "        else:\n"
            "            image_embeds, deepstack_image_embeds = _image_out\n",
        )
        text = text.replace(
            "        video_embeds, deepstack_video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)\n",
            "        _video_out = model.visual(pixel_values_videos, grid_thw=video_grid_thw)\n"
            "        if hasattr(_video_out, 'pooler_output'):\n"
            "            video_embeds = _video_out.pooler_output\n"
            "            deepstack_video_embeds = _video_out.deepstack_features\n"
            "        else:\n"
            "            video_embeds, deepstack_video_embeds = _video_out\n",
        )
        text = text.replace(
            "        image_embeds, dummy_deepstack_image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)\n",
            "        _dummy_out = model.visual(pixel_values, grid_thw=image_grid_thw)\n"
            "        if hasattr(_dummy_out, 'pooler_output'):\n"
            "            image_embeds = _dummy_out.pooler_output\n"
            "            dummy_deepstack_image_embeds = _dummy_out.deepstack_features\n"
            "        else:\n"
            "            image_embeds, dummy_deepstack_image_embeds = _dummy_out\n",
        )
    elif path.name == "tensordict_utils.py":
        needle = "        assert isinstance(val, torch.Tensor | list)\n"
        inject = (
            "        if not isinstance(val, (torch.Tensor, list, NonTensorStack)):\n"
            "            tensor_dict[key] = NonTensorData(val)\n"
            "            continue\n\n"
            "        assert isinstance(val, torch.Tensor | list)\n"
        )
        text = text.replace(needle, inject)

    if text != original:
        path.write_text(text, encoding="utf-8")
PY

python "${ROOT_DIR}/fine_tuning/grpo_prep.py" \
  --metadata "${ROOT_DIR}/data/train/metadata.jsonl" \
  --dataset-root "${ROOT_DIR}/data/train" \
  --output-dir "${ROOT_DIR}/fine_tuning/data"

if ! python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "ERROR: CUDA is not available in this environment; Qwen3-VL-4B GRPO training requires at least one GPU."
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES

python -m verl.trainer.main_ppo \
  actor_rollout_ref.model.path=Qwen/Qwen3-VL-4B-Instruct \
  actor_rollout_ref.model.tokenizer_path=Qwen/Qwen3-VL-4B-Instruct \
  actor_rollout_ref.model.trust_remote_code=true \
  +actor_rollout_ref.model.override_config._attn_implementation=eager \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.load_format=hf \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  critic.enable=false \
  data.train_files="${ROOT_DIR}/fine_tuning/data/morse_train_grpo_train.json" \
  data.val_files="${ROOT_DIR}/fine_tuning/data/morse_train_grpo_val.json" \
  data.prompt_key=prompt \
  data.reward_fn_key=data_source \
  data.max_prompt_length=4096 \
  data.max_response_length=256 \
  data.train_batch_size=2 \
  data.val_batch_size=2 \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts=false \
  data.return_raw_chat=true \
  data.return_multi_modal_inputs=true \
  custom_reward_function.path="${ROOT_DIR}/fine_tuning/grpo_reward.py" \
  custom_reward_function.name=compute_score \
  trainer.project_name=morse500 \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger='[console]' \
  trainer.val_before_train=false \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1
