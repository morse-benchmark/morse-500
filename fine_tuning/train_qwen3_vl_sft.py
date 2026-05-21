#!/usr/bin/env python3
"""LoRA SFT for Qwen3-VL-4B using local video QA data.

Expected metadata JSONL fields:
- file_name: relative path under --dataset-root to an MP4
- question_text: user question
- solution: target answer text

Example:
python fine_tuning/train_qwen3_vl_sft.py \
  --metadata data/train/metadata.jsonl \
  --dataset-root data/train \
  --output-dir fine_tuning/checkpoints/qwen3-vl-4b-lora
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

try:
    from transformers import AutoModelForVision2Seq as _QWEN_VL_AUTO_MODEL
except ImportError:
    try:
        from transformers import AutoModelForImageTextToText as _QWEN_VL_AUTO_MODEL
    except ImportError as exc:
        raise ImportError(
            "Your transformers version is too old for Qwen3-VL fine-tuning. "
            "Please upgrade with: pip install -U 'transformers>=4.45.0'"
        ) from exc


SYSTEM_PROMPT = (
    "You are a helpful visual reasoning assistant. "
    "Answer the question using only evidence from the video."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--metadata", default="data/train/metadata.jsonl")
    parser.add_argument("--dataset-root", default="data/train")
    parser.add_argument("--output-dir", default="fine_tuning/checkpoints/qwen3-vl-4b-lora")
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def load_samples(metadata_path: Path, dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            rel_video = _safe_text(row.get("file_name"))
            question = _safe_text(row.get("question_text"))
            answer = _safe_text(row.get("solution"))
            if not rel_video or not question or not answer:
                continue
            video_path = (dataset_root / rel_video).resolve()
            if not video_path.exists():
                continue

            category = _safe_text(row.get("category"))
            prompt = question if not category else f"[Category: {category}] {question}"
            sample_id = Path(rel_video).stem or f"sample_{idx}"

            rows.append(
                {
                    "id": sample_id,
                    "video": str(video_path),
                    "question": prompt,
                    "answer": answer,
                    "messages": [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "video", "path": str(video_path)},
                                {"type": "text", "text": prompt},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": answer}],
                        },
                    ],
                }
            )
    return rows


def build_datasets(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    if not rows:
        raise ValueError("No valid rows found in metadata.")

    rng = random.Random(seed)
    rng.shuffle(rows)

    n_val = int(len(rows) * val_ratio)
    if n_val <= 0:
        n_val = 1
    if n_val >= len(rows):
        n_val = max(1, len(rows) - 1)

    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    return Dataset.from_list(train_rows), Dataset.from_list(val_rows)


def _collect_media(messages: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    image_inputs: list[str] = []
    video_inputs: list[str] = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "image":
                uri = part.get("path") or part.get("image")
                if uri:
                    image_inputs.append(str(uri))
            elif ptype == "video":
                uri = part.get("path") or part.get("video")
                if uri:
                    video_inputs.append(str(uri))
    return image_inputs, video_inputs


@dataclass
class VLMDataCollator:
    processor: Any
    max_pixels: int
    fps: float

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[list[str]] = []
        videos: list[list[str]] = []

        for ex in examples:
            msgs = ex["messages"]
            text = self.processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
            img_paths, vid_paths = _collect_media(msgs)
            images.append(img_paths)
            videos.append(vid_paths)

        # For video-only rows, images must be None (not an empty nested list).
        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "videos": videos,
            "return_tensors": "pt",
            "padding": True,
            "max_pixels": self.max_pixels,
            "fps": self.fps,
        }
        if any(len(sample_images) > 0 for sample_images in images):
            processor_kwargs["images"] = images

        # Qwen3-VL processor accepts nested image/video path lists.
        batch = self.processor(**processor_kwargs)

        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        for token_attr in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ):
            token_id = getattr(self.processor.tokenizer, token_attr, None)
            if token_id is None:
                token_id = getattr(getattr(self.processor, "tokenizer", object()), token_attr, None)
            if token_id is not None:
                labels[labels == token_id] = -100

        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 or --fp16")

    torch.manual_seed(args.seed)

    metadata_path = Path(args.metadata)
    dataset_root = Path(args.dataset_root)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    rows = load_samples(metadata_path, dataset_root)
    train_ds, val_ds = build_datasets(rows, args.val_ratio, args.seed)

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    model = _QWEN_VL_AUTO_MODEL.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: peft. Install with `pip install peft` or run "
            "`bash fine_tuning/run_qwen3_vl_sft.sh`."
        ) from exc

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        bf16=args.bf16,
        fp16=args.fp16,
        seed=args.seed,
    )

    collator = VLMDataCollator(
        processor=processor,
        max_pixels=args.max_pixels,
        fps=args.fps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()

    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
