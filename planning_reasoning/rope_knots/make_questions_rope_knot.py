import os
import random
import re
import numpy as np
import moviepy.editor as mpy
from moviepy.editor import transfx, CompositeVideoClip
import moviepy.video.fx.all as vfx
from PIL import Image, ImageDraw, ImageFont
import tempfile
import cv2
import argparse
import json

def create_question_knot_reorder(image_folder, output_video, task_name="knot_reorder", difficulty=5, slide_duration=3, with_trans_effect=False):
    """
    Create a slideshow with smooth sliding transitions for a knot tying quiz.
    
    Args:
        image_folder: Folder containing the knot tying sequence images
        output_video: Path for the output video file
        task_name: Name of the task (e.g., "knot_reorder")
        difficulty: Number of images to use (difficulty level)
        slide_duration: Duration to show each slide in seconds
        with_trans_effect: Whether to apply transition effects
    """
    # Get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Extract image numbers from filenames (assuming they have R1, R2, etc.)
    numbered_files = []
    for img_file in image_files:
        # Look for R followed by a number in the filename
        match = re.search(r'R(\d+)', img_file)
        if match:
            number = int(match.group(1))
            numbered_files.append((number, img_file))
    
    # Sort by the extracted numbers
    numbered_files.sort()
    
    # Extract just the filenames in the correct order
    all_files = [f for _, f in numbered_files]
    
    # Limit to difficulty level (number of images)
    if difficulty > len(all_files):
        print(f"Warning: Requested difficulty {difficulty} exceeds available images {len(all_files)}. Using all available images.")
        difficulty = len(all_files)
    
    # Select subset based on difficulty level
    if difficulty < len(all_files):
        # Evenly space out the selected images across the sequence
        indices = np.linspace(0, len(all_files) - 1, difficulty, dtype=int)
        correct_order = [all_files[i] for i in indices]
    else:
        correct_order = all_files
    
    # Create a shuffled version of the sequence
    shuffled_files = correct_order.copy()
    random.shuffle(shuffled_files)
    
    # Map shuffled positions to their correct sequence
    correct_sequence = []
    for file in correct_order:
        idx = shuffled_files.index(file)
        correct_sequence.append(str(idx+1))

    solution = ",".join(correct_sequence)
    print(f"Task: {task_name}, Difficulty: {difficulty}")
    print("Correct sequence:", correct_order)
    print("Shuffled sequence:", shuffled_files)
    print("Solution:", solution)
    
    # Create temporary directory for our slides
    temp_dir = tempfile.mkdtemp()
    
    # Get dimensions from the first image to ensure consistency
    first_img_path = os.path.join(image_folder, shuffled_files[0])
    with Image.open(first_img_path) as img:
        img_width, img_height = img.size
    
    knot_name = image_folder.split('/')[-1].replace('_',' ')
    # Create the instruction slide
    instruction_img_path = os.path.join(temp_dir, "instruction.jpg")
    
    rephrased_questions = [
        "The frames displayed are shuffled clips from tying {knot_name}. What is the correct chronological order of tying this knot from start to end?",
        "These clips show steps from tying {knot_name} in random sequence – arrange them in their proper creation order.",
        "The displayed frames are scrambled segments of {knot_name} being tied. What's the actual progression from beginning to completion?",
        "You're viewing mixed-up video clips from the process of tying {knot_name}. Reconstruct the original step sequence.",
        "These shuffled video segments depict {knot_name} being tied. Put them back in chronological formation.",
        "The frames shown are jumbled clips from {knot_name} knot-tying. Reorder them into the correct timeline.",
        "Randomized clips of {knot_name} being tied are displayed. Identify the correct start-to-end sequence.",
        "These scrambled frames capture stages of tying {knot_name}. Restore their logical progression.",
        "Out-of-order clips from the {knot_name} tying process are shown. Determine their proper temporal arrangement.",
        "The video segments of {knot_name} being tied are disordered. Sort them into the right step-by-step order.",
        "You're seeing randomly organized footage of {knot_name} knot-tying. What's the true chronological workflow?",
        "These mixed-up frames document {knot_name} being tied. Sequence them from initiation to completion.",
        "Shuffled clips from the {knot_name} tying process are displayed. Map them to their correct positions in the timeline.",
        "The presented clips show {knot_name} knot steps in disorder. Reassemble them in their authentic progression.",
        "Jumbled video segments of {knot_name} being tied are shown. What sequence represents the actual tying process?",
        "These randomized frames display {knot_name} knot-tying stages. Organize them into their genuine formation order.",
        "Scrambled footage from tying {knot_name} is visible. Piece together the proper execution sequence.",
        "The clips exhibit {knot_name} tying steps in chaotic order. Reestablish their correct procedural flow.",
        "Disordered frames from {knot_name} knot creation are shown. What arrangement reflects the real construction process?",
        "You're viewing arbitrarily ordered clips of {knot_name} being tied. Replicate the authentic step progression.",
        "These perturbed video segments show {knot_name} knot-tying. Deduce their original temporal configuration."
    ]
    instruction_text = random.choice(rephrased_questions).format(knot_name=knot_name)
    hint_text = f"Only output a comma-separated number sequence (e.g., 1,2,3)."
    create_instruction_slide(instruction_img_path, 
                            instruction_text,
                            hint_text,
                            img_width, img_height,
                            f"The correct sequence is: {solution}")
    
    # Prepare all image clips
    clips = []
    transition_duration = 0.8  # Duration of the slide transition in seconds
    
    # Add each labeled image
    for i, img_file in enumerate(shuffled_files):
        # Create a labeled version of the image
        img_path = os.path.join(image_folder, img_file)
        labeled_img_path = os.path.join(temp_dir, f"labeled_{i}.jpg")
        
        label = str(i+1)
        add_label_to_image(img_path, labeled_img_path, label)
        
        clip = mpy.ImageClip(labeled_img_path).set_duration(slide_duration)
        clips.append(clip)
    
    # Add instruction slide
    instruction_clip = mpy.ImageClip(instruction_img_path).set_duration(slide_duration * 2)  # Longer duration
    clips.append(instruction_clip)
    
    # Apply fading transitions to all clips (except the first one)
    final_clips = [clips[0]]  # First clip doesn't need a transition

    if not with_trans_effect:
        for i in range(1, len(clips)):
            # Apply crossfadein for smooth transitions
            clip_with_fx = clips[i].crossfadein(transition_duration)
            final_clips.append(clip_with_fx)
    else:
        transition_types = ["slide", "rotate", "zoom_in", "blur_in", "color_fade", "wave", "pixelate"]
        def wave_in(clip, duration):
            def effect(get_frame, t):
                if t < duration:
                    frame = get_frame(t)
                    h, w = frame.shape[:2]
                    
                    # Create proper format for the maps
                    progress = t/duration
                    strength = (1-progress) * 30
                    
                    # Create mesh grid
                    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
                    
                    # Apply wave distortion
                    map_y = map_y + strength * np.sin(map_x/30 + progress*6)
                    
                    # Convert to proper format for cv2.remap
                    map_x = map_x.astype(np.float32)
                    map_y = map_y.astype(np.float32)
                    
                    # Apply distortion (ensuring maps are in the correct format)
                    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
                else:
                    return get_frame(t)
            return clip.fl(effect)
        
        def pixelate_in(clip, duration):
            def effect(get_frame, t):
                if t < duration:
                    # Calculate pixel size (starts large, gets smaller)
                    pixel_size = int(max(1, 50 * (1 - t/duration)))
                    frame = get_frame(t)
                    h, w = frame.shape[:2]
                    
                    # Reduce resolution
                    small = cv2.resize(frame, (w//pixel_size, h//pixel_size))
                    # Bring back to original size (with big pixels)
                    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    return get_frame(t)
            return clip.fl(effect)
        
        def slide_in(clip, duration, direction="right"):
            """Slide in transition effect."""
            def effect(get_frame, t):
                if t < duration:
                    frame = get_frame(t)
                    h, w = frame.shape[:2]
                    
                    # Create a blank canvas (black background)
                    result = np.zeros_like(frame)
                    
                    # Calculate progress (0 to 1)
                    progress = t / duration
                    
                    # Calculate position based on direction
                    if direction == "right":
                        # Slide in from right to left
                        offset = int((1 - progress) * w)
                        # Copy visible portion of the frame
                        result[:, :max(0, w-offset)] = frame[:, offset:]
                    elif direction == "left":
                        # Slide in from left to right
                        offset = int((1 - progress) * w)
                        # Copy visible portion of the frame
                        result[:, offset:] = frame[:, :max(0, w-offset)]
                    elif direction == "top":
                        # Slide in from top to bottom
                        offset = int((1 - progress) * h)
                        # Copy visible portion of the frame
                        result[offset:, :] = frame[:max(0, h-offset), :]
                    elif direction == "bottom":
                        # Slide in from bottom to top
                        offset = int((1 - progress) * h)
                        # Copy visible portion of the frame
                        result[:max(0, h-offset), :] = frame[offset:, :]
                    
                    return result
                else:
                    return get_frame(t)
            
            return clip.fl(effect)

        def zoom_in(clip, duration, start_scale=0.2):
            """Zoom in transition effect."""
            def effect(get_frame, t):
                if t < duration:
                    frame = get_frame(t)
                    h, w = frame.shape[:2]
                    
                    # Calculate progress (0 to 1)
                    progress = t / duration
                    
                    # Calculate scale factor (starts small, ends at 1.0)
                    # Ensure minimum scale is 0.1 to avoid resize errors
                    scale = max(0.1, start_scale + (1 - start_scale) * progress)
                    
                    # Calculate dimensions
                    new_h = int(h * scale)
                    new_w = int(w * scale)
                    
                    # Ensure minimum dimensions of 1 pixel
                    new_h = max(1, new_h)
                    new_w = max(1, new_w)
                    
                    # Resize frame to smaller size
                    small = cv2.resize(frame, (new_w, new_h))
                    
                    # Create a blank canvas (black background)
                    result = np.zeros_like(frame)
                    
                    # Calculate offsets to center the resized image
                    y_offset = (h - new_h) // 2
                    x_offset = (w - new_w) // 2
                    
                    # Place the resized image in the center of the canvas
                    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = small
                    
                    return result
                else:
                    return get_frame(t)
            
            return clip.fl(effect)

        def blur_in(clip, duration, max_blur=30):
            """Blur in transition effect."""
            def effect(get_frame, t):
                if t < duration:
                    frame = get_frame(t)
                    
                    # Calculate progress (0 to 1)
                    progress = t / duration
                    
                    # Calculate blur kernel size (starts large, ends at 1)
                    # Blur kernel must be odd and >= 1
                    blur_size = int(max(1, max_blur * (1 - progress)))
                    if blur_size % 2 == 0:  # Ensure odd kernel size
                        blur_size += 1
                    
                    # Apply gaussian blur
                    if blur_size > 1:  # Only apply blur if kernel size > 1
                        return cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
                    else:
                        return frame
                else:
                    return get_frame(t)
            
            return clip.fl(effect)
        
        for i in range(1, len(clips)):
            current_clip = clips[i]
            transition_type = transition_types[i % len(transition_types)]
            
            if transition_type == "slide":
                clip_with_fx = slide_in(current_clip, transition_duration)
            if transition_type == "rotate":
                clip_with_fx = clips[i].fx(vfx.rotate, lambda t: 360*(1-min(1, t/transition_duration)))
            elif transition_type == "zoom_in":
                clip_with_fx = zoom_in(current_clip, transition_duration)
            elif transition_type == "blur_in":
                clip_with_fx = blur_in(current_clip, transition_duration)
            elif transition_type == "color_fade":
                # Start from grayscale to color
                clip_with_fx = current_clip.fx(vfx.blackwhite).fx(vfx.fadein, transition_duration)
            elif transition_type == "wave":
                clip_with_fx = wave_in(current_clip, transition_duration)
            elif transition_type == "pixelate":
                clip_with_fx = pixelate_in(current_clip, transition_duration)
            else:
                clip_with_fx = current_clip.crossfadein(transition_duration)
                
            # You can still add crossfade on top of these effects for smoother transitions
            clip_with_fx = clip_with_fx.crossfadein(transition_duration/2)
            final_clips.append(clip_with_fx)

    # Concatenate all clips
    final_clip = mpy.concatenate_videoclips(final_clips, method="compose")
    
    # Write the final video
    final_clip.write_videofile(output_video, fps=24)
    
    print(f"Slideshow created: {output_video}")
    print(f"The correct sequence is: {solution}")
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    
    question_text = f"{instruction_text}\n{hint_text}"
    return solution, question_text


def create_question_knot_remaining_order(image_folder, output_video, task_name="knot_remaining_order", difficulty=5, slide_duration=3):
    """
    Create a slideshow with smooth sliding transitions for a knot tying quiz.
    
    Args:
        image_folder: Folder containing the knot tying sequence images
        output_video: Path for the output video file
        task_name: Name of the task
        difficulty: Number of images to use (difficulty level)
        slide_duration: Duration to show each slide in seconds
    """
    # Get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Extract image numbers from filenames (assuming they have R1, R2, etc.)
    numbered_files = []
    for img_file in image_files:
        # Look for R followed by a number in the filename
        match = re.search(r'R(\d+)', img_file)
        if match:
            number = int(match.group(1))
            numbered_files.append((number, img_file))
    
    # Sort by the extracted numbers
    numbered_files.sort()
    
    # Extract just the filenames in the correct order
    all_files = [f for _, f in numbered_files]
    
    # Limit to difficulty level (number of images)
    if difficulty > len(all_files):
        print(f"Warning: Requested difficulty {difficulty} exceeds available images {len(all_files)}. Using all available images.")
        difficulty = len(all_files)
    
    # Select subset based on difficulty level
    if difficulty < len(all_files):
        # Evenly space out the selected images across the sequence
        indices = np.linspace(0, len(all_files) - 1, difficulty, dtype=int)
        correct_order = [all_files[i] for i in indices]
    else:
        correct_order = all_files
    
    # Create a shuffled version of the sequence
    shuffled_files = correct_order.copy()
    random.shuffle(shuffled_files)
    
    # Map shuffled positions to their correct sequence
    correct_sequence = []
    for file in correct_order:
        idx = shuffled_files.index(file)
        correct_sequence.append(str(idx+1))

    random_step = random.randint(1, len(correct_order))
    random_step_idx = correct_sequence.index(str(random_step))
    solution = ",".join(correct_sequence[random_step_idx+1:])

    print(f"Task: {task_name}, Difficulty: {difficulty}")
    print("Correct sequence:", correct_order)
    print("Shuffled sequence:", shuffled_files)
    print("Solution:", solution)
    
    # Create temporary directory for our slides
    temp_dir = tempfile.mkdtemp()
    
    # Get dimensions from the first image to ensure consistency
    first_img_path = os.path.join(image_folder, shuffled_files[0])
    with Image.open(first_img_path) as img:
        img_width, img_height = img.size
    
    knot_name = image_folder.split('/')[-1].replace('_',' ')
    # Create the instruction slide
    instruction_img_path = os.path.join(temp_dir, "instruction.jpg")
    question_templates = [
        "This is step {current} of tying {knot_name}. What are the remaining steps needed to complete the knot, listed in chronological order?",
        "You're at step {current} in tying {knot_name}. What are the remaining steps required to finish, in chronological order?",
        "When tying {knot_name}, you've reached step {current}. What are the remaining steps, listed in chronological order?",
        "Looking at step {current} in tying {knot_name}, what are the remaining steps needed to complete it, in chronological order?",
        "This is step {current} of the {knot_name} knot. What are the remaining steps you need to perform, listed in chronological order?",
        "You're now at step {current} of {knot_name}. What are the remaining steps needed after this point, in chronological order?",
        "At step {current} of tying {knot_name}, what are the remaining moves you must make to finish, listed in chronological order?",
        "Step {current} shown in tying {knot_name}. What are the remaining steps before completion, in chronological order?",
        "If you've completed step {current} while tying {knot_name}, what are the remaining actions you need to perform, listed in chronological order?"
    ]
    
    # Choose a random question template
    instruction_text = random.choice(question_templates).format(
        knot_name=knot_name,
        current=random_step
    )
    hint_text = f"Only output a comma-separated number sequence (e.g., 1,2,3)."
    create_instruction_slide(instruction_img_path, 
                            instruction_text,
                            hint_text,
                            img_width, img_height,
                            f"The correct sequence is: {solution}")
    
    # Prepare all image clips
    clips = []
    transition_duration = 0.8  # Duration of the slide transition in seconds
    
    # Add each labeled image
    for i, img_file in enumerate(shuffled_files):
        # Create a labeled version of the image
        img_path = os.path.join(image_folder, img_file)
        labeled_img_path = os.path.join(temp_dir, f"labeled_{i}.jpg")
        
        label = str(i+1)
        add_label_to_image(img_path, labeled_img_path, label)
        
        clip = mpy.ImageClip(labeled_img_path).set_duration(slide_duration)
        clips.append(clip)
    
    # Add instruction slide
    instruction_clip = mpy.ImageClip(instruction_img_path).set_duration(slide_duration * 2)  # Longer duration
    clips.append(instruction_clip)
    
    # Apply fading transitions to all clips (except the first one)
    final_clips = [clips[0]]  # First clip doesn't need a transition

    for i in range(1, len(clips)):
        # Apply crossfadein for smooth transitions
        clip_with_fx = clips[i].crossfadein(transition_duration)
        final_clips.append(clip_with_fx)

    # Concatenate all clips
    final_clip = mpy.concatenate_videoclips(final_clips, method="compose")
    
    # Write the final video
    final_clip.write_videofile(output_video, fps=24)
    
    print(f"Slideshow created: {output_video}")
    print(f"The correct sequence is: {solution}")
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    
    question_text = f"{instruction_text}\n{hint_text}"
    return solution, question_text


def create_question_knot_remaining_count(image_folder, output_video, task_name="knot_remaining_count", difficulty=5, slide_duration=3):
    """
    Create a slideshow with smooth sliding transitions for a knot tying quiz.
    
    Args:
        image_folder: Folder containing the knot tying sequence images
        output_video: Path for the output video file
        task_name: Name of the task
        difficulty: Number of images to use (difficulty level)
        slide_duration: Duration to show each slide in seconds
    """
    # Get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Extract image numbers from filenames (assuming they have R1, R2, etc.)
    numbered_files = []
    for img_file in image_files:
        # Look for R followed by a number in the filename
        match = re.search(r'R(\d+)', img_file)
        if match:
            number = int(match.group(1))
            numbered_files.append((number, img_file))
    
    # Sort by the extracted numbers
    numbered_files.sort()
    
    # Extract just the filenames in the correct order
    all_files = [f for _, f in numbered_files]
    
    # Limit to difficulty level (number of images)
    if difficulty > len(all_files):
        print(f"Warning: Requested difficulty {difficulty} exceeds available images {len(all_files)}. Using all available images.")
        difficulty = len(all_files)
    
    # Select subset based on difficulty level
    if difficulty < len(all_files):
        # Evenly space out the selected images across the sequence
        indices = np.linspace(0, len(all_files) - 1, difficulty, dtype=int)
        correct_order = [all_files[i] for i in indices]
    else:
        correct_order = all_files
    
    # Create a shuffled version of the sequence
    shuffled_files = correct_order.copy()
    random.shuffle(shuffled_files)
    
    # Map shuffled positions to their correct sequence
    correct_sequence = []
    for file in correct_order:
        idx = shuffled_files.index(file)
        correct_sequence.append(str(idx+1))

    random_step = random.randint(1, len(correct_order))
    random_step_idx = correct_sequence.index(str(random_step))
    remaining_steps = len(correct_sequence) - random_step_idx - 1
    solution = remaining_steps
    
    print(f"Task: {task_name}, Difficulty: {difficulty}")
    print("Correct sequence:", correct_order)
    print("Shuffled sequence:", shuffled_files)
    print("Solution:", solution)
    
    # Create temporary directory for our slides
    temp_dir = tempfile.mkdtemp()
    
    # Get dimensions from the first image to ensure consistency
    first_img_path = os.path.join(image_folder, shuffled_files[0])
    with Image.open(first_img_path) as img:
        img_width, img_height = img.size
    
    knot_name = image_folder.split('/')[-1].replace('_',' ')
    # Create the instruction slide
    instruction_img_path = os.path.join(temp_dir, "instruction.jpg")
    question_templates = [
        "This is step {current} of tying {knot_name}. How many more steps are needed to complete the knot?",
        "You're at step {current} in tying {knot_name}. How many additional steps are required to finish?",
        "When tying {knot_name}, you've reached step {current}. How many more steps remain?",
        "Looking at step {current} in tying {knot_name}, how many further steps are needed to complete it?",
        "This is step {current} of the {knot_name} knot. How many more steps do you need to perform?",
        "You're now at step {current} of {knot_name}. Count how many additional steps are needed after this point.",
        "At step {current} of tying {knot_name}, how many more moves must you make to finish?",
        "Step {current} shown in tying {knot_name}. How many steps remain before completion?",
        "If you've completed step {current} while tying {knot_name}, how many more actions do you need to perform?"
    ]
    
    # Choose a random question template
    instruction_text = random.choice(question_templates).format(
        knot_name=knot_name,
        current=random_step
    )
    hint_text = f"Your answer should only be a number (e.g. 1)."
    create_instruction_slide(instruction_img_path, 
                            instruction_text,
                            hint_text,
                            img_width, img_height,
                            f"The correct answer is: {solution}")
    
    # Prepare all image clips
    clips = []
    transition_duration = 0.8  # Duration of the slide transition in seconds
    
    # Add each labeled image
    for i, img_file in enumerate(shuffled_files):
        # Create a labeled version of the image
        img_path = os.path.join(image_folder, img_file)
        labeled_img_path = os.path.join(temp_dir, f"labeled_{i}.jpg")
        
        label = str(i+1)
        add_label_to_image(img_path, labeled_img_path, label)
        
        clip = mpy.ImageClip(labeled_img_path).set_duration(slide_duration)
        clips.append(clip)
    
    # Add instruction slide
    instruction_clip = mpy.ImageClip(instruction_img_path).set_duration(slide_duration * 2)  # Longer duration
    clips.append(instruction_clip)
    
    # Apply fading transitions to all clips (except the first one)
    final_clips = [clips[0]]  # First clip doesn't need a transition

    for i in range(1, len(clips)):
        # Apply crossfadein for smooth transitions
        clip_with_fx = clips[i].crossfadein(transition_duration)
        final_clips.append(clip_with_fx)
        
    # Concatenate all clips
    final_clip = mpy.concatenate_videoclips(final_clips, method="compose")
    
    # Write the final video
    final_clip.write_videofile(output_video, fps=24)
    
    print(f"Slideshow created: {output_video}")
    print(f"The correct answer is: {solution}")
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    
    question_text = f"{instruction_text}\n{hint_text}"
    return solution, question_text


def create_instruction_slide(output_path, instruction_text, hint_text, width, height, solution_text=None):
    """Create an image with the instruction text using the same dimensions as the input images"""
    # Create a blank image with the same dimensions as input images
    background_color = (240, 240, 240)
    
    image = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(image)
    font_size = int(height * 0.05)  # 5% of image height
    font = ImageFont.load_default(font_size)
    solution_font = ImageFont.load_default(font_size)
    
    # Draw the instruction
    text_color = (30, 30, 30)
    
    # Handle long instruction text by splitting into lines if needed
    words = instruction_text.split()
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        line_text = " ".join(current_line)
        line_width = draw.textlength(line_text, font=font)
        
        if line_width > width * 0.8:  # If line would be too wide
            current_line.pop()  # Remove the last word
            lines.append(" ".join(current_line))  # Add the line without the last word
            current_line = [word]  # Start a new line with the last word
    
    # Add the last line
    if current_line:
        lines.append(" ".join(current_line))
    
    # Draw each line of the instruction
    y_position = height//2 - (len(lines) * (font_size + 10)) // 2
    
    for line in lines:
        line_width = draw.textlength(line, font=font)
        x_position = width//2 - line_width//2  # Center horizontally
        
        draw.text((x_position, y_position), line, fill=text_color, font=font)
        y_position += font_size + 10  # Move down for next line
    
    # Add hint text
    hint_width = draw.textlength(hint_text, font=font)
    hint_position = (width//2 - hint_width//2, y_position + 10)
    
    hint_color = (200, 100, 100)
    draw.text(hint_position, hint_text, fill=hint_color, font=font)
    
    # Save the image
    image.save(output_path)


def add_label_to_image(input_path, output_path, label):
    """Add a letter label to an image"""
    # Open the image
    image = Image.open(input_path)
    width, height = image.size
    min_side = min(width, height)
    
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    font_size = int(height * 0.1)  # 10% of image height
    font = ImageFont.load_default(font_size)
    
    circle_position = (int(min_side * 0.1), int(min_side * 0.1))  # Positioned at 10% from top-left
    
    # Calculate text position to center it in the circle
    text_width = draw.textlength(label, font=font)
    text_position = (circle_position[0] - text_width//2, 
                     circle_position[1] - font_size//2)
    
    # Draw text
    draw.text(text_position, label, fill=(255, 0, 0), font=font)
    
    # Save the image
    image.save(output_path)


def get_difficulty_mapping():
    """Define difficulty levels and their corresponding number of images"""
    return {
        "easy": 3,
        "medium": 5,
        "hard": 8,
        "expert": 12
    }


def parse_task_configs(args):
    """Parse command line arguments to create task configurations"""
    task_configs = []
    
    # If config file is provided, load from JSON
    if args.config:
        try:
            with open(args.config, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading config file: {e}")
            return []
    
    # Build configurations from command line arguments
    if args.tasks:
        for task in args.tasks:
            config = {"task_name": task}
            
            # Set difficulty levels
            if args.difficulties:
                config["difficulty_levels"] = args.difficulties
            else:
                # Default difficulty levels based on task
                if task in ["knot_reorder", "knot_remaining_order"]:
                    config["difficulty_levels"] = ["easy", "medium", "hard", "expert"]
                elif task == "knot_reorder_effect":
                    config["difficulty_levels"] = ["medium", "hard"]
                else:  # knot_remaining_count
                    config["difficulty_levels"] = ["easy", "medium", "hard"]
            
            # Set effects flag
            if task == "knot_reorder" and args.with_effects:
                config["with_effects"] = True
            elif task == "knot_reorder_effect":
                config["with_effects"] = True
            else:
                config["with_effects"] = False
                
            task_configs.append(config)
    
    return task_configs


def main():
    parser = argparse.ArgumentParser(description='Generate knot tying video questions')
    
    # Input/Output directories
    parser.add_argument('--input-dir', default='AnimatedKnots', 
                       help='Directory containing knot image folders (default: AnimatedKnots)')
    parser.add_argument('--output-dir', default='questions', 
                       help='Output directory for videos (default: questions)')
    parser.add_argument('--question-text-dir', default='question_text',
                       help='Directory for question text files (default: question_text)')
    parser.add_argument('--solution-dir', default='solutions',
                       help='Directory for solution files (default: solutions)')
    
    # Task configuration options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=str,
                      help='JSON file containing task configurations')
    group.add_argument('--tasks', nargs='+', 
                      choices=['knot_reorder', 'knot_reorder_effect', 'knot_remaining_order', 'knot_remaining_count'],
                      help='List of tasks to run')
    
    # Additional task options
    parser.add_argument('--difficulties', nargs='+',
                       choices=['easy', 'medium', 'hard', 'expert'],
                       help='Difficulty levels to generate (default: varies by task)')
    parser.add_argument('--with-effects', action='store_true',
                       help='Apply transition effects to knot_reorder tasks')
    
    # Processing options
    parser.add_argument('--slide-duration', type=float, default=1.5,
                       help='Duration for each slide in seconds (default: 1.5)')
    parser.add_argument('--max-knots', type=int, default=None,
                       help='Maximum number of knots to process (default: all)')
    
    args = parser.parse_args()
    
    # Parse task configurations
    task_configs = parse_task_configs(args)
    
    if not task_configs:
        print("No valid task configurations found. Please check your arguments or config file.")
        return
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.question_text_dir, exist_ok=True) 
    os.makedirs(args.solution_dir, exist_ok=True)
    
    # Define difficulty mapping
    difficulty_map = get_difficulty_mapping()
    
    print(f"Task configurations:")
    for config in task_configs:
        print(f"  - {config}")
    print()
    
    # Process each knot
    knot_dirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    
    if args.max_knots:
        knot_dirs = knot_dirs[:args.max_knots]
    
    total_tasks = len(knot_dirs) * sum(len(config.get("difficulty_levels", [])) for config in task_configs)
    current_task = 0
    
    for idx, knot_name in enumerate(knot_dirs):
        print(f"Processing knot {idx+1}/{len(knot_dirs)}: {knot_name}")
        
        for config in task_configs:
            task_name = config["task_name"]
            
            for difficulty_level in config.get("difficulty_levels", []):
                current_task += 1
                difficulty = difficulty_map[difficulty_level]
                
                # Folder containing the knot sequence images
                image_folder = os.path.join(args.input_dir, knot_name)
                
                # Create output filename with difficulty level
                output_video = os.path.join(args.output_dir, f"{task_name}_{difficulty_level}_{knot_name}.mp4")
                
                print(f"  Task {current_task}/{total_tasks}: {task_name} - {difficulty_level}")
                
                # Create the slideshow based on task type
                try:
                    if task_name == "knot_reorder" or task_name == "knot_reorder_effect":
                        solution, question_text = create_question_knot_reorder(
                            image_folder, output_video, 
                            task_name=task_name, 
                            difficulty=difficulty, 
                            slide_duration=args.slide_duration, 
                            with_trans_effect=config.get("with_effects", False)
                        )
                    elif task_name == "knot_remaining_order":
                        solution, question_text = create_question_knot_remaining_order(
                            image_folder, output_video, 
                            task_name=task_name, 
                            difficulty=difficulty, 
                            slide_duration=args.slide_duration
                        )
                    elif task_name == "knot_remaining_count":
                        solution, question_text = create_question_knot_remaining_count(
                            image_folder, output_video, 
                            task_name=task_name, 
                            difficulty=difficulty, 
                            slide_duration=args.slide_duration
                        )
                    
                    # Save question text and solution with difficulty level
                    question_filename = f"{task_name}_{difficulty_level}_{knot_name}.txt"
                    
                    with open(os.path.join(args.question_text_dir, question_filename), "w") as f:
                        f.write(question_text)
                    
                    with open(os.path.join(args.solution_dir, question_filename), "w") as f:
                        f.write(str(solution))
                        
                    print(f"    ✓ Completed: {task_name} - {difficulty_level} - {knot_name}")
                    
                except Exception as e:
                    print(f"    ✗ Error processing {knot_name} with {task_name} at {difficulty_level}: {e}")
                    continue
    
    print(f"\nCompleted processing {len(knot_dirs)} knots with {len(task_configs)} task configurations.")


if __name__ == "__main__":
    main()