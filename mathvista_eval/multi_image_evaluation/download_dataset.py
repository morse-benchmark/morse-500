import os
import numpy as np
import pickle
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def create_question_slides(question, width, height):
    """Split question across multiple slides if needed"""
    background_color = (240, 240, 240)
    font_size = max(int(height * 0.05), 10)
    font = ImageFont.load_default(font_size)

    text_color = (30, 30, 30)
    max_line_width = width * 0.9
    max_lines_per_slide = int((height * 0.8) // (font_size + 10))

    # Split question into words
    words = question.split()
    slides = []
    current_slide_lines = []
    current_line = []

    for word in words:
        current_line.append(word)
        test_line = " ".join(current_line)
        line_width = ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(
            test_line, font=font
        )

        if line_width > max_line_width:
            current_line.pop()
            if current_line:
                current_slide_lines.append(" ".join(current_line))
            current_line = [word]

            if len(current_slide_lines) >= max_lines_per_slide:
                slides.append(current_slide_lines)
                current_slide_lines = []

    # Add remaining lines
    if current_line:
        current_slide_lines.append(" ".join(current_line))
    if current_slide_lines:
        slides.append(current_slide_lines)

    # Create PIL Images for each slide
    slide_images = []
    for i, slide_lines in enumerate(slides):
        img = Image.new("RGBA", (width, height), background_color)
        draw = ImageDraw.Draw(img)

        # Calculate starting y position
        total_text_height = len(slide_lines) * (font_size + 10)
        y_position = (height - total_text_height) // 2

        for line in slide_lines:
            line_width = draw.textlength(line, font=font)
            x_position = (width - line_width) // 2
            draw.text((x_position, y_position), line, fill=text_color, font=font)
            y_position += font_size + 10

        slide_images.append(img)

    return slide_images


def save_image_sequence(image, question, output_dir, pid):
    """Save original image and question slides as separate images"""
    os.makedirs(output_dir, exist_ok=True)

    # Save original image
    original_path = os.path.join(output_dir, f"{pid}_original.jpg")
    image.convert("RGB").save(original_path)

    # Create and save question slides
    question_slides = create_question_slides(question, image.width, image.height)
    slide_paths = []

    for i, slide in enumerate(question_slides):
        slide_path = os.path.join(output_dir, f"{pid}_question_{i}.jpg")
        slide.convert("RGB").save(slide_path)
        slide_paths.append(slide_path)

    return [original_path] + slide_paths


def main():
    # Process MathVista dataset
    dataset = load_dataset("AI4Math/MathVista", split="testmini")
    output_dir = "mathvista_images"
    os.makedirs(output_dir, exist_ok=True)

    output_data = []
    for example in tqdm(dataset, desc="Processing MathVista"):
        pid = example["pid"]
        question = example["query"]
        image = example["decoded_image"]
        answer = example.get("answer", None)

        image_paths = save_image_sequence(image, question, output_dir, pid)

        if image_paths:
            output_data.append(
                {
                    "pid": pid,
                    "image_paths": image_paths,
                    "question": question,
                    "answer": answer,
                }
            )

    # Save dataset metadata
    with open("mathvista_image_dataset.pkl", "wb") as f:
        pickle.dump(output_data, f)

    print(f"Saved {len(output_data)} image sequences to {output_dir}")


if __name__ == "__main__":
    main()
