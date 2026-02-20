import os
import argparse
import subprocess
from pathlib import Path
from typing import List

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------


def get_video_dimensions(video_path):
    """Get video dimensions using ffprobe"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        width, height = map(int, result.stdout.strip().split(","))
        return width, height
    except Exception:
        return 0, 0


def resize_video_ffmpeg(video_path, max_size=512):
    """Resize video using FFmpeg. Returns path to temp file or None."""
    try:
        width, height = get_video_dimensions(video_path)
        if width == 0 or height == 0:
            return None

        if width > height:
            new_width = min(max_size, width)
            new_height = int(height * (new_width / width))
        else:
            new_height = min(max_size, height)
            new_width = int(width * (new_height / height))

        # Ensure even dimensions
        new_width = new_width if new_width % 2 == 0 else new_width - 1
        new_height = new_height if new_height % 2 == 0 else new_height - 1

        if new_width >= width and new_height >= height:
            return None

        temp_dir = Path("/tmp/vllm_video_cache")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"resized_{Path(video_path).stem}_{os.getpid()}.mp4"

        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"scale={new_width}:{new_height}",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-preset",
            "fast",
            "-an",
            "-y",
            str(temp_path),
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return str(temp_path)
    except Exception as e:
        print(f"Error resizing {video_path}: {e}")
        return None


def get_video_files(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return []
    extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    files = []
    for ext in extensions:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted([str(f) for f in files])


# -------------------------------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------------------------------


def process_batch(llm, tokenizer, video_paths, output_dir, query, params, max_size):
    inputs = []
    meta = []
    temp_files = []

    for path in video_paths:
        stem = Path(path).stem
        if (output_dir / f"{stem}.txt").exists():
            continue

        # Resize
        resized = resize_video_ffmpeg(path, max_size)
        final_path = resized if resized else path
        if resized:
            temp_files.append(resized)

        # Prepare Chat
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": final_path},
                    {"type": "text", "text": query},
                ],
            }
        ]

        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

        inputs.append({"prompt": prompt, "multi_modal_data": {"video": final_path}})
        meta.append(stem)

    if not inputs:
        return

    try:
        outputs = llm.generate(inputs, sampling_params=params)
        for i, out in enumerate(outputs):
            with open(output_dir / f"{meta[i]}.txt", "w", encoding="utf-8") as f:
                f.write(out.outputs[0].text)
    except Exception as e:
        print(f"Batch failed: {e}")
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output_dir", type=str, default=".")

    # GPU & Performance Args
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="ID of the GPU to use (e.g., '0' or '0,1')",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to split the model across",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_video_size", type=int, default=512)
    parser.add_argument("--gpu_utilization", type=float, default=0.95)

    args = parser.parse_args()

    # 1. Set Visible Devices BEFORE importing torch/vllm internals heavily
    # This forces the script to only see the specific GPU ID provided
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print(f"--- Config ---")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"Model: {args.model_name}")

    # Setup Output
    out_path = Path(args.output_dir) / args.model_name.split("/")[-1]
    out_path.mkdir(parents=True, exist_ok=True)

    # Load Videos
    videos = get_video_files(args.video_folder)
    if not videos:
        return

    # Init LLM
    print("Loading Model...")
    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,  # <--- KEY CHANGE 1
        gpu_memory_utilization=args.gpu_utilization,
        max_model_len=32768,
        limit_mm_per_prompt={"video": 1},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=2048)
    query = "Answer the question in this video. Show reasoning step-by-step, then put final answer in \\boxed{}."

    # Loop
    with tqdm(total=len(videos)) as pbar:
        for i in range(0, len(videos), args.batch_size):
            batch = videos[i : i + args.batch_size]
            process_batch(
                llm,
                tokenizer,
                batch,
                out_path,
                query,
                sampling_params,
                args.max_video_size,
            )
            pbar.update(len(batch))


if __name__ == "__main__":
    main()
