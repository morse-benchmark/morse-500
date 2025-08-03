import os
import cv2
import numpy as np
import pickle
import re
import random
import tempfile
import shutil
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import moviepy.editor as mpy
from moviepy.editor import transfx, CompositeVideoClip
import moviepy.video.fx.all as vfx


def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format"""
    rgb_image = pil_image.convert("RGB")
    return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)


def create_question_slides(question, width, height, temp_dir):
    """Split question across multiple slides if needed"""
    background_color = (240, 240, 240)
    font_size = max(int(height * 0.05), 10)
    font = ImageFont.load_default(font_size)

    text_color = (30, 30, 30)
    max_line_width = width * 0.9
    max_lines_per_slide = int(
        (height * 0.8) // (font_size + 10)
    )  # Calculate max lines that fit

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
            # Move last word to new line
            current_line.pop()
            if current_line:
                current_slide_lines.append(" ".join(current_line))
            current_line = [word]

            # Check if we need a new slide
            if len(current_slide_lines) >= max_lines_per_slide:
                slides.append(current_slide_lines)
                current_slide_lines = []

    # Add remaining lines
    if current_line:
        current_slide_lines.append(" ".join(current_line))
    if current_slide_lines:
        slides.append(current_slide_lines)

    # Create image files for each slide
    slide_paths = []
    for i, slide_lines in enumerate(slides):
        img = Image.new("RGB", (width, height), background_color)
        draw = ImageDraw.Draw(img)

        # Calculate starting y position to center the text vertically
        total_text_height = len(slide_lines) * (font_size + 10)
        y_position = (height - total_text_height) // 2

        for line in slide_lines:
            line_width = draw.textlength(line, font=font)
            x_position = (width - line_width) // 2
            draw.text((x_position, y_position), line, fill=text_color, font=font)
            y_position += font_size + 10

        # Add page indicator if multiple slides
        slide_path = os.path.join(temp_dir, f"question_{i}.jpg")
        img.save(slide_path)
        slide_paths.append(slide_path)

    return slide_paths


def create_question_video(
    image, question, output_path, duration=2, fps=24, resize_shape=None
):
    """Create a video showing the image followed by the question (split across frames if needed)"""
    temp_dir = None
    try:
        # Convert the image to OpenCV format
        frame = pil_to_cv2(image)

        # Keep original dimensions if resize_shape is not specified
        if resize_shape is None:
            h, w, _ = frame.shape
        else:
            frame = cv2.resize(frame, resize_shape)
            h, w, _ = frame.shape

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        # Create question slides with same dimensions as the image
        question_frames = create_question_slides(question, w, h, temp_dir)

        # Calculate frame counts
        image_frames = duration * fps
        question_frames_count = (
            len(question_frames) * fps
        )  # Show question slides for 1 second each

        # Use MP4V codec for MP4 format
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        if not video_writer.isOpened():
            raise Exception("Could not open video writer")

        # Write image frames
        for _ in range(image_frames):
            video_writer.write(frame)

        # Write question frames (cycling through each slide)
        for i in range(question_frames_count):
            frame_idx = i // fps  # Determine which slide to show
            question_frame = cv2.imread(question_frames[frame_idx])
            video_writer.write(question_frame)

        video_writer.release()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create video at {output_path}: {e}")
        return False
    finally:
        # Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def create_slideshow(image_folder, output_video, slide_duration=3, fps=24):
    """Create a slideshow with smooth transitions from images in a folder"""
    try:
        # Get all image files in the folder
        image_files = [
            f
            for f in os.listdir(image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Extract image numbers from filenames (assuming R1, R2, etc.)
        numbered_files = []
        for img_file in image_files:
            match = re.search(r"R(\d+)", img_file)
            if match:
                number = int(match.group(1))
                numbered_files.append((number, img_file))

        # Sort by the extracted numbers
        numbered_files.sort()
        sorted_files = [f[1] for f in numbered_files]

        # Get dimensions from the first image
        first_img_path = os.path.join(image_folder, sorted_files[0])
        with Image.open(first_img_path) as img:
            img_width, img_height = img.size

        # Prepare image clips
        clips = []
        for img_file in sorted_files:
            img_path = os.path.join(image_folder, img_file)
            clip = mpy.ImageClip(img_path).set_duration(slide_duration)
            clips.append(clip)

        # Apply transitions between clips
        final_clips = [clips[0]]
        transition_duration = 0.8

        for i in range(1, len(clips)):
            transition = transfx.crossfadein(transition_duration)
            final_clips.append(
                clips[i]
                .set_start(final_clips[-1].end - transition_duration)
                .crossfadein(transition_duration)
            )

        # Concatenate all clips
        final_clip = mpy.concatenate_videoclips(final_clips, method="compose")

        # Write the final video
        final_clip.write_videofile(
            output_video, fps=fps, codec="libx264", audio_codec="aac"
        )
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create slideshow: {e}")
        return False


def main():
    # Process MathVista dataset
    dataset = load_dataset("AI4Math/MathVista", split="testmini")
    os.makedirs("mathvista_videos", exist_ok=True)

    output_data = []
    for example in tqdm(dataset, desc="Processing MathVista"):
        pid = example["pid"]
        question = example["query"]
        image = example["decoded_image"]
        answer = example.get("answer", None)

        video_path = f"mathvista_videos/{pid}.mp4"
        success = create_question_video(image, question, video_path)

        if success:
            output_data.append(
                {"video": video_path, "question": question, "answer": answer}
            )

    # Save dataset metadata
    with open("mathvista_video_dataset.pkl", "wb") as f:
        pickle.dump(output_data, f)

    # Process slideshow folders (example usage)
    slideshow_dir = "slideshow_images"
    output_dir = "slideshow_videos"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(slideshow_dir):
        for folder in tqdm(os.listdir(slideshow_dir), desc="Processing slideshows"):
            folder_path = os.path.join(slideshow_dir, folder)
            if os.path.isdir(folder_path):
                output_video = os.path.join(output_dir, f"{folder}.mp4")
                create_slideshow(folder_path, output_video)


if __name__ == "__main__":
    main()
