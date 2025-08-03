import pickle
import csv
import os
import numpy as np
import cv2
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# Constants
MIN_IMAGE_DIM = 28  # Minimum dimension required by Qwen2.5-VL
BATCH_SIZE = 1000  # Adjust based on GPU memory
TARGET_FPS = 1  # Frames per second to extract


def validate_and_resize_image(image):
    """Ensure image meets minimum size requirements"""
    width, height = image.size
    if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
        # Resize while maintaining aspect ratio
        ratio = max(MIN_IMAGE_DIM / width, MIN_IMAGE_DIM / height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def validate_and_resize_frame(frame: np.ndarray) -> np.ndarray:
    """Ensure video frame meets minimum size requirements"""
    height, width = frame.shape[:2]
    if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
        # Resize while maintaining aspect ratio
        ratio = max(MIN_IMAGE_DIM / width, MIN_IMAGE_DIM / height)
        new_size = (int(width * ratio), int(height * ratio))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_LANCZOS4)
    return frame


def load_video_frames(video_path: str) -> np.ndarray:
    """Load and validate video frames as numpy array"""
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
            # Convert to RGB and validate size
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            validated_frame = validate_and_resize_frame(rgb_frame)
            frames.append(validated_frame)
        frame_count += 1

    cap.release()
    return np.array(frames)


def process_batch(batch_items, llm, processor):
    """Process a batch of videos in parallel"""
    try:
        # Load and validate all video frames in parallel
        with ThreadPoolExecutor() as executor:
            frames = list(
                executor.map(load_video_frames, [item["video"] for item in batch_items])
            )

            # Prepare all prompts in parallel
            prompts = list(
                executor.map(
                    lambda item: processor.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "video",
                                        "video": f"file://{os.path.abspath(item['video'])}",
                                        "max_pixels": 360 * 420,
                                        "fps": TARGET_FPS,
                                    },
                                    {
                                        "type": "text",
                                        "text": "Answer the question in the video.",
                                    },
                                ],
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    ),
                    batch_items,
                )
            )

        # Prepare batch inputs for vLLM
        batch_inputs = [
            {"prompt": prompt, "multi_modal_data": {"video": frame_sequence}}
            for prompt, frame_sequence in zip(prompts, frames)
        ]

        # Process all videos in a single batch
        outputs = llm.generate(
            batch_inputs,
            sampling_params=SamplingParams(
                max_tokens=1024, temperature=0, top_p=1.0, stop=["<|im_end|>"]
            ),
        )

        # Extract responses
        return [
            output.outputs[0].text.strip().split("assistant")[-1].strip()
            for output in outputs
        ]

    except Exception as e:
        print(f"Error processing batch: {e}")
        return ["Error processing video"] * len(batch_items)


def main():
    # Load dataset
    with open("mathvista_video_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)

    # Initialize vLLM with numpy array support
    print("Loading Qwen2.5-VL model with vLLM...")
    llm = LLM(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        dtype="bfloat16",
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

    # Process videos in parallel batches
    results = []
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Processing videos"):
        batch_items = dataset[i : i + BATCH_SIZE]
        predictions = process_batch(batch_items, llm, processor)

        for j, prediction in enumerate(predictions):
            results.append(
                {
                    "video_id": os.path.splitext(
                        os.path.basename(batch_items[j]["video"])
                    )[0],
                    "ground_truth": str(batch_items[j]["answer"]),
                    "prediction": prediction,
                }
            )

        # Save periodic checkpoints
        if i % (BATCH_SIZE * 10) == 0 and i > 0:
            with open("MathVista_QWen2.5_VL_checkpoint.csv", "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["video_id", "ground_truth", "prediction"]
                )
                writer.writeheader()
                writer.writerows(results)

    # Save final results
    csv_file = "MathVista_QWen2.5_VL-7B_Instruct_Video.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_id", "ground_truth", "prediction"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_file}")


if __name__ == "__main__":
    main()
