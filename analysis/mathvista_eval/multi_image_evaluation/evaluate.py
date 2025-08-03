import pickle
import csv
import os
import numpy as np
import cv2
import statistics
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from concurrent.futures import ThreadPoolExecutor
from Levenshtein import distance
from datasets import load_dataset
from PIL import Image

# Constants
MIN_IMAGE_DIM = 28
TARGET_FPS = 1


def validate_and_resize_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
        ratio = max(MIN_IMAGE_DIM / width, MIN_IMAGE_DIM / height)
        new_size = (int(width * ratio), int(height * ratio))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_LANCZOS4)
    return frame


def load_video_frames(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(original_fps / TARGET_FPS))
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            validated = validate_and_resize_frame(rgb_frame)
            frames.append(validated)
        frame_count += 1

    cap.release()
    return np.array(frames)


def calculate_similarity(original, extracted):
    if not extracted or extracted.startswith("Error"):
        return 0.0
    max_len = max(len(original), len(extracted))
    if max_len == 0:
        return 1.0
    return 1 - (distance(original.lower(), extracted.lower()) / max_len)


def process_batch(batch_items, llm, processor, reference_dataset):
    results = []
    similarity_scores = []

    try:
        with ThreadPoolExecutor() as executor:
            frames = list(
                executor.map(load_video_frames, [item["video"] for item in batch_items])
            )

        prompts = []
        for item in batch_items:
            print(os.path.abspath(item["video"]))
            prompt = processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": "file://" + os.path.abspath(item["video"]),
                                "max_pixels": 360 * 420,
                                "fps": TARGET_FPS,
                            },
                            {
                                "type": "text",
                                "text": "Extract the full query text verbatim at the end of the video.",
                            },
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        batch_inputs = [
            {"prompt": prompt, "multi_modal_data": {"video": frame_sequence}}
            for prompt, frame_sequence in zip(prompts, frames)
        ]

        outputs = llm.generate(
            batch_inputs,
            sampling_params=SamplingParams(
                max_tokens=512,
                temperature=0,
                top_p=1.0,
                stop=["<|im_end|>"],
            ),
        )

        for i, output in enumerate(outputs):
            extracted = output.outputs[0].text.strip().split("assistant")[-1].strip()
            video_id = os.path.splitext(os.path.basename(batch_items[i]["video"]))[0]

            # Match with MathVista ground truth
            gt_question = reference_dataset[i]["question"]
            sim = calculate_similarity(gt_question, extracted)
            is_correct = sim >= 0.9

            results.append(
                {
                    "video_id": video_id,
                    "ground_truth": gt_question,
                    "prediction": extracted,
                    "similarity": round(sim, 3),
                    "is_correct": is_correct,
                }
            )
            similarity_scores.append(sim)

        return results, similarity_scores

    except Exception as e:
        print(f"Error during batch processing: {e}")
        return [], []


def main():
    with open("../video_evaluation/mathvista_video_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    mathvista_dataset = load_dataset("AI4Math/MathVista", split="testmini")

    print("Loading Qwen2.5-VL model with vLLM...")
    llm = LLM(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        dtype="float16",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        mm_processor_kwargs={
            "min_pixels": MIN_IMAGE_DIM * MIN_IMAGE_DIM,
            "max_pixels": 360 * 420,
            "accept_numpy": True,
        },
    )

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        max_pixels=360 * 420,
    )

    print(f"Processing {len(dataset)} videos in a single batch...")
    results, similarity_scores = process_batch(
        dataset, llm, processor, mathvista_dataset
    )

    if results:
        accuracy = sum(1 for r in results if r["is_correct"]) / len(results)
        avg_similarity = statistics.mean(similarity_scores)
        median_similarity = statistics.median(similarity_scores)

        output_file = "question_extraction_full_batch_vllm.csv"
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"\nExtraction Complete (processed {len(results)} videos)")
        print(f"Accuracy (90%+ match): {accuracy:.2%}")
        print(f"Average similarity: {avg_similarity:.2f}")
        print(f"Median similarity: {median_similarity:.2f}")
        print(f"Results saved to {output_file}")
    else:
        print("No results were generated due to errors")


if __name__ == "__main__":
    main()
