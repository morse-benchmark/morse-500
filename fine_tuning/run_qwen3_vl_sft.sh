#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"

if [[ ! -d "${ROOT_DIR}/.morse_venv" ]]; then
  python3 -m venv "${ROOT_DIR}/.morse_venv"
fi
source "${ROOT_DIR}/.morse_venv/bin/activate"

python -m pip install --upgrade pip
pip install -U \
  "torch>=2.2" torchvision torchaudio \
  "transformers==5.2.0" datasets pandas pyarrow \
  peft accelerate bitsandbytes qwen-vl-utils av decord \
  verl

# VERL 0.7 expects AutoModelForVision2Seq. Transformers 5.x renamed this API.
# Patch installed VERL to add backward-compatible aliases.
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
        # transformers 5.x returns BaseModelOutputWithDeepstackFeatures for visual tower.
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

python "${ROOT_DIR}/fine_tuning/verl_sft_prep.py" \
  --metadata "${ROOT_DIR}/data/train/metadata.jsonl" \
  --dataset-root "${ROOT_DIR}/data/train" \
  --output-dir "${ROOT_DIR}/fine_tuning/data"

export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES

torchrun --standalone --nnodes=1 --nproc_per_node=1 -m verl.trainer.sft_trainer \
  model.path=Qwen/Qwen3-VL-4B-Instruct \
  model.tokenizer_path=Qwen/Qwen3-VL-4B-Instruct \
  model.trust_remote_code=true \
  +model.override_config._attn_implementation=eager \
  model.enable_gradient_checkpointing=true \
  model.lora_rank=16 \
  model.lora_alpha=32 \
  model.use_remove_padding=false \
  engine.use_torch_compile=false \
  data.train_files="${ROOT_DIR}/fine_tuning/data/morse_train_sft_train.parquet" \
  data.val_files="${ROOT_DIR}/fine_tuning/data/morse_train_sft_val.parquet" \
  data.messages_key=messages \
  +data.video_key=videos \
  data.pad_mode=no_padding \
  data.max_length=4096 \
  data.train_batch_size=1 \
  data.micro_batch_size_per_gpu=1 \
  data.max_token_len_per_gpu=4096 \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.use_dynamic_bsz=false \
  trainer.project_name=morse500 \
  trainer.experiment_name=qwen3-vl-4b-verl-sft \
  trainer.logger='[console]' \
  trainer.total_epochs=1 \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1
