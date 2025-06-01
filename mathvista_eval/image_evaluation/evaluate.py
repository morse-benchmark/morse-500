import csv
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

BATCH_SIZE = 1000
MIN_IMAGE_DIM = 28  # Minimum dimension required by Qwen2.5-VL


def validate_and_resize_image(image):
    """Ensure image meets minimum size requirements"""
    width, height = image.size
    if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
        # Resize while maintaining aspect ratio
        ratio = max(MIN_IMAGE_DIM / width, MIN_IMAGE_DIM / height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def generate_predictions(batch, llm, sampling_params, processor):
    prompts = []
    multi_modal_data = []
    for i in range(len(batch["query"])):
        # Validate and resize image if needed
        image = validate_and_resize_image(batch["decoded_image"][i].convert("RGB"))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": batch["query"][i]},
                    {"type": "image", "image": image},
                ],
            }
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        prompts.append(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            }
        )

    try:
        outputs = llm.generate(prompts, sampling_params)
        predictions = [output.outputs[0].text for output in outputs]
        return predictions
    except Exception as e:
        print(f"Generation error: {e}")
        return [""] * len(prompts)


def main():
    dataset = load_dataset("AI4Math/MathVista", split="testmini")
    sampling_params = SamplingParams(
        max_tokens=1024, temperature=0, top_p=1.0, stop=["<|im_end|>"]
    )

    # Initialize LLM with proper image processing constraints
    llm = LLM(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        max_model_len=4096,
        mm_processor_kwargs={
            "min_pixels": MIN_IMAGE_DIM * MIN_IMAGE_DIM,
            "max_pixels": 360 * 420,
        },
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        max_pixels=360 * 420,
    )

    results = []
    total_items = min(1000, len(dataset))  # Ensure we don't exceed dataset size

    for i in tqdm(range(0, total_items, BATCH_SIZE), desc="Evaluating images"):
        end_idx = min(i + BATCH_SIZE, total_items)
        batch = {
            "query": dataset["query"][i:end_idx],
            "pid": dataset["pid"][i:end_idx],
            "decoded_image": dataset["decoded_image"][i:end_idx],
            "answer": dataset["answer"][i:end_idx],
        }

        predictions = generate_predictions(batch, llm, sampling_params, processor)

        for j, prediction in enumerate(predictions):
            results.append(
                {
                    "image_id": batch["pid"][j],
                    "ground_truth": batch["answer"][j],
                    "prediction": prediction,
                }
            )

    csv_file = "MathVista_Qwen2.5-VL-7B-Instruct_Image.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_id", "ground_truth", "prediction"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {csv_file}")


if __name__ == "__main__":
    main()
