#!/usr/bin/env python3
"""Offline batched vLLM inference for Qwen3-VL on data/train metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a helpful visual reasoning assistant. "
    "Answer the question using only evidence from the video."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--metadata", default="data/train/metadata.jsonl")
    parser.add_argument("--dataset-root", default="data/train")
    parser.add_argument("--output", default="fine_tuning/checkpoints/vllm_batch_predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=8192)
    return parser.parse_args()


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def load_rows(metadata_path: Path, dataset_root: Path, max_samples: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            rel_video = _safe_text(row.get("file_name"))
            question = _safe_text(row.get("question_text"))
            answer = _safe_text(row.get("solution"))
            if not rel_video or not question:
                continue
            video_path = (dataset_root / rel_video).resolve()
            if not video_path.exists():
                continue
            sample_id = Path(rel_video).stem or f"sample_{idx}"
            rows.append(
                {
                    "id": sample_id,
                    "video_path": str(video_path),
                    "question": question,
                    "answer": answer,
                }
            )
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def batched(items: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    out: list[list[dict[str, str]]] = []
    for i in range(0, len(items), batch_size):
        out.append(items[i : i + batch_size])
    return out


def build_prompt(processor: Any, question: str) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": question},
            ],
        },
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    args = parse_args()

    try:
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies. Install with: pip install -U vllm transformers"
        ) from exc

    metadata_path = Path(args.metadata)
    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(metadata_path, dataset_root, args.max_samples)
    if not rows:
        raise SystemExit("No valid rows found.")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"video": 1},
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    with output_path.open("w", encoding="utf-8") as fout:
        for chunk in batched(rows, args.batch_size):
            requests = []
            for row in chunk:
                prompt = build_prompt(processor, row["question"])
                requests.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"video": row["video_path"]},
                    }
                )

            outputs = llm.generate(requests, sampling_params=sampling_params)
            for row, output in zip(chunk, outputs):
                pred = output.outputs[0].text.strip() if output.outputs else ""
                rec = {
                    "id": row["id"],
                    "video_path": row["video_path"],
                    "question": row["question"],
                    "target": row["answer"],
                    "prediction": pred,
                }
                fout.write(json.dumps(rec, ensure_ascii=True) + "\n")

    print(f"Wrote {len(rows)} predictions to {output_path}")


if __name__ == "__main__":
    main()
