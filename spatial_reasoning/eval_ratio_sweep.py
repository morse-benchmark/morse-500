#!/usr/bin/env python3
"""
Ratio Sweep Evaluation System
==============================

Evaluates models across multiple size ratios to determine:
1. Accuracy at each ratio level
2. Highest ratio achieved before failure
3. Step size for difficulty progression

Uses vLLM for efficient parallel inference.

Usage:
    python eval_ratio_sweep.py --model Qwen/Qwen3-VL-8B-Instruct --port 8000 \\
        --program cubes --ratios "1,2,5,10,15,20,25,30" --samples_per_ratio 10
"""

import os
import sys
import json
import base64
import asyncio
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import re


# ============================================================================
# Video Processing Utilities
# ============================================================================

def get_video_dimensions(video_path):
    """Get video dimensions using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    width, height = map(int, result.stdout.strip().split(','))
    return width, height


def resize_video_ffmpeg(video_path, max_size=512):
    """Resize video using FFmpeg if needed"""
    try:
        width, height = get_video_dimensions(video_path)

        if width > height:
            new_width = min(max_size, width)
            new_height = int(height * (new_width / width))
        else:
            new_height = min(max_size, height)
            new_width = int(width * (new_height / height))

        new_width = new_width if new_width % 2 == 0 else new_width - 1
        new_height = new_height if new_height % 2 == 0 else new_height - 1

        if new_width >= width and new_height >= height:
            return None

        temp_path = f"/tmp/temp_resized_{Path(video_path).stem}_{os.getpid()}.mp4"

        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'scale={new_width}:{new_height}',
            '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
            '-an', '-y', temp_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return None

        return temp_path

    except Exception as e:
        print(f"Error resizing video: {e}")
        return None


def encode_b64(file_path, max_size=None):
    """Encode file to base64 with optional resizing"""
    file_ext = Path(file_path).suffix.lower()
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    if max_size and file_ext in video_extensions:
        temp_path = None
        try:
            temp_path = resize_video_ffmpeg(file_path, max_size=max_size)
            file_to_encode = temp_path if temp_path else file_path

            with open(file_to_encode, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")

            return encoded
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    else:
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")


# ============================================================================
# Rate Limiting
# ============================================================================

class AsyncRateLimiter:
    def __init__(self, calls_per_minute):
        self.calls_per_minute = calls_per_minute
        self.period_seconds = 60
        self.calls_timestamps = deque()
        self.lock = asyncio.Lock()

    async def wait_if_needed(self):
        async with self.lock:
            now = datetime.now()

            while self.calls_timestamps and self.calls_timestamps[0] < now - timedelta(seconds=self.period_seconds):
                self.calls_timestamps.popleft()

            if len(self.calls_timestamps) >= self.calls_per_minute:
                oldest_timestamp = self.calls_timestamps[0]
                wait_time = (oldest_timestamp + timedelta(seconds=self.period_seconds) - now).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            self.calls_timestamps.append(datetime.now())


# ============================================================================
# Answer Extraction
# ============================================================================

def extract_answer(response_text: str, answer_type: str = "integer") -> str:
    """
    Extract answer from model response.

    Looks for \\boxed{} format first, then falls back to extracting
    the last number/word depending on answer_type.
    """
    if not response_text:
        return None

    # Try to extract from \\boxed{}
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, response_text)
    if boxed_match:
        answer = boxed_match.group(1).strip()
        if answer_type == "integer":
            # Extract just the number
            num_match = re.search(r'\d+', answer)
            return num_match.group(0) if num_match else answer
        return answer.lower()

    # Fallback: extract last occurrence
    if answer_type == "integer":
        # Find all numbers in the response
        numbers = re.findall(r'\b\d+\b', response_text)
        return numbers[-1] if numbers else None
    else:
        # For color names, look for common colors at end
        color_pattern = r'\b(red|blue|green|yellow|purple|orange|teal|pink|white)\b'
        matches = re.findall(color_pattern, response_text.lower())
        return matches[-1] if matches else None


# ============================================================================
# Model Querying
# ============================================================================

async def query_video(client, model_name, video_path, query, rate_limiter,
                     max_size=512, max_retries=3, retry_delay=2):
    """Query a video with the model"""

    for attempt in range(max_retries):
        try:
            await rate_limiter.wait_if_needed()

            try:
                base64_video = encode_b64(video_path, max_size=max_size)
                video_url = f"data:video/mp4;base64,{base64_video}"
            except Exception as e:
                print(f"Error encoding video {video_path}: {e}")
                return None

            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": query},
                                {"type": "video_url", "video_url": {"url": video_url}},
                            ],
                        }
                    ],
                )
            except Exception as e:
                print(f"API request failed for {video_path}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return None

            if response is None or not hasattr(response, 'choices') or not response.choices:
                return None

            choice = response.choices[0]
            if not hasattr(choice, 'message') or not hasattr(choice.message, 'content'):
                return ""

            return choice.message.content

        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed for {video_path}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

    return None


# ============================================================================
# Generation and Evaluation
# ============================================================================

def generate_videos_at_ratio(program: str, ratio: float, p_type: str,
                            num_samples: int, output_dir: Path):
    """Generate videos at a specific ratio level"""
    print(f"\n[GENERATE] Generating {num_samples} videos for {program} at ratio={ratio}%")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check how many already exist
    existing = list(output_dir.glob(f"questions/{program}_ratio_{p_type}_r{ratio:.1f}_*.mp4"))
    if len(existing) >= num_samples:
        print(f"  ✓ {len(existing)} videos already exist, skipping generation")
        return existing[:num_samples]

    # Generate missing videos
    needed = num_samples - len(existing)
    print(f"  Generating {needed} new videos...")

    for i in range(needed):
        env = os.environ.copy()
        env['SIZE_RATIO'] = str(ratio)
        env['P_TYPE'] = p_type

        try:
            result = subprocess.run(
                ['python', f'{program}_ratio.py'],
                cwd=str(output_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                print(f"  ✗ Failed to generate video {i+1}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout generating video {i+1}")
        except Exception as e:
            print(f"  ✗ Error generating video {i+1}: {e}")

    # Return all videos at this ratio
    all_videos = list(output_dir.glob(f"questions/{program}_ratio_{p_type}_r{ratio:.1f}_*.mp4"))
    return all_videos[:num_samples]


async def evaluate_video(client, model_name, video_path: Path, solution_path: Path,
                        query: str, rate_limiter, answer_type: str, max_size: int):
    """Evaluate a single video"""

    # Get ground truth
    if not solution_path.exists():
        return None, None, "no_solution"

    with open(solution_path, 'r') as f:
        ground_truth = f.read().strip().lower()

    # Query model
    response = await query_video(client, model_name, str(video_path), query,
                                rate_limiter, max_size=max_size)

    if response is None:
        return ground_truth, None, "no_response"

    # Extract answer
    predicted = extract_answer(response, answer_type)

    if predicted is None:
        return ground_truth, None, "no_answer_extracted"

    # Check correctness
    predicted = predicted.lower()
    is_correct = (predicted == ground_truth)

    return ground_truth, predicted, "correct" if is_correct else "incorrect"


async def evaluate_ratio_level(client, model_name, video_paths: List[Path],
                               query: str, rate_limiter, answer_type: str,
                               max_size: int, max_concurrent: int) -> Dict:
    """Evaluate all videos at one ratio level"""

    semaphore = asyncio.Semaphore(max_concurrent)

    async def eval_with_semaphore(video_path):
        async with semaphore:
            # Find solution file
            solution_path = video_path.parent.parent / "solutions" / video_path.name.replace('.mp4', '.txt')
            return await evaluate_video(client, model_name, video_path, solution_path,
                                       query, rate_limiter, answer_type, max_size)

    tasks = [eval_with_semaphore(vp) for vp in video_paths]
    results = []

    for task in tqdm.as_completed(tasks, total=len(tasks), desc="Evaluating"):
        result = await task
        results.append(result)

    # Calculate metrics
    correct = sum(1 for _, _, status in results if status == "correct")
    incorrect = sum(1 for _, _, status in results if status == "incorrect")
    no_response = sum(1 for _, _, status in results if status == "no_response")
    no_answer = sum(1 for _, _, status in results if status == "no_answer_extracted")

    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "no_response": no_response,
        "no_answer_extracted": no_answer,
        "accuracy": accuracy,
        "results": results
    }


# ============================================================================
# Main Sweep
# ============================================================================

async def run_ratio_sweep(args):
    """Run complete ratio sweep evaluation"""

    # Parse ratios
    ratios = [float(r.strip()) for r in args.ratios.split(',')]
    ratios.sort()

    print(f"{'='*80}")
    print(f"Ratio Sweep Evaluation")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Program: {args.program}")
    print(f"Problem Type: {args.p_type}")
    print(f"Ratios: {ratios}")
    print(f"Samples per ratio: {args.samples_per_ratio}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"{'='*80}\n")

    # Setup client
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{args.port}/v1",
    )

    rate_limiter = AsyncRateLimiter(calls_per_minute=args.rate_limit)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine answer type
    answer_type = "integer" if args.p_type in ["count", "missing", "surface_area", "exposed", "colors", "project"] else "color"

    # Query template
    query = "Answer the question in this video. Show your reasoning step-by-step, then put your final answer in \\boxed{}."

    # Results storage
    all_results = {}

    # Process each ratio
    for ratio in ratios:
        print(f"\n{'='*80}")
        print(f"Processing SIZE_RATIO = {ratio}%")
        print(f"{'='*80}")

        # Generate videos
        video_paths = generate_videos_at_ratio(
            args.program, ratio, args.p_type,
            args.samples_per_ratio, output_dir
        )

        if not video_paths:
            print(f"  ✗ No videos generated for ratio={ratio}%")
            continue

        print(f"  ✓ {len(video_paths)} videos ready")

        # Evaluate
        print(f"  Evaluating with {args.model}...")
        metrics = await evaluate_ratio_level(
            client, args.model, video_paths, query, rate_limiter,
            answer_type, args.max_size, args.max_concurrent
        )

        all_results[ratio] = metrics

        print(f"\n  Results for ratio={ratio}%:")
        print(f"    Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
        print(f"    Incorrect: {metrics['incorrect']}")
        print(f"    No response: {metrics['no_response']}")
        print(f"    No answer extracted: {metrics['no_answer_extracted']}")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Ratio':<10} {'Accuracy':<12} {'Correct':<10} {'Total':<10}")
    print(f"{'-'*42}")

    for ratio in ratios:
        if ratio in all_results:
            m = all_results[ratio]
            print(f"{ratio:<10.1f} {m['accuracy']:<12.2%} {m['correct']:<10} {m['total']:<10}")

    # Find threshold (last ratio with >90% accuracy)
    threshold_ratio = None
    for ratio in ratios:
        if ratio in all_results and all_results[ratio]['accuracy'] >= 0.9:
            threshold_ratio = ratio
        else:
            break

    if threshold_ratio:
        print(f"\n✓ Highest ratio with ≥90% accuracy: {threshold_ratio}%")
    else:
        print(f"\n✗ No ratio achieved ≥90% accuracy")

    # Save results
    results_file = output_dir / f"results_{args.program}_{args.p_type}_{args.model.replace('/', '_')}.json"
    with open(results_file, 'w') as f:
        # Convert results to serializable format
        serializable_results = {}
        for ratio, metrics in all_results.items():
            serializable_results[str(ratio)] = {
                "total": metrics["total"],
                "correct": metrics["correct"],
                "incorrect": metrics["incorrect"],
                "no_response": metrics["no_response"],
                "no_answer_extracted": metrics["no_answer_extracted"],
                "accuracy": metrics["accuracy"]
            }

        json.dump({
            "model": args.model,
            "program": args.program,
            "p_type": args.p_type,
            "ratios": ratios,
            "samples_per_ratio": args.samples_per_ratio,
            "threshold_ratio": threshold_ratio,
            "results": serializable_results
        }, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")

    await client.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ratio sweep evaluation system")
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-8B-Instruct',
                       help='Model name for vLLM')
    parser.add_argument('--port', type=int, default=8000,
                       help='vLLM server port')
    parser.add_argument('--program', type=str, required=True,
                       choices=['cubes', 'ropes', 'domino_count'],
                       help='Which program to evaluate')
    parser.add_argument('--p_type', type=str, default='count',
                       help='Problem type (count, missing, colors, etc.)')
    parser.add_argument('--ratios', type=str, default='1,2,5,10,15,20,25,30',
                       help='Comma-separated list of ratios to test')
    parser.add_argument('--samples_per_ratio', type=int, default=10,
                       help='Number of samples to generate per ratio')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for videos and results')
    parser.add_argument('--max_concurrent', type=int, default=5,
                       help='Maximum concurrent API calls')
    parser.add_argument('--max_size', type=int, default=512,
                       help='Maximum video dimension')
    parser.add_argument('--rate_limit', type=int, default=20,
                       help='API calls per minute')

    args = parser.parse_args()

    asyncio.run(run_ratio_sweep(args))


if __name__ == "__main__":
    main()
