import csv
from pathlib import Path
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip
import textwrap


def get_video_fps_moviepy(video_path):
    clip = VideoFileClip(video_path)
    fps = clip.fps
    duration = clip.duration
    width, height = clip.size
    clip.close()
    return fps, duration, width, height


def add_letter_to_video(video_clip, letter):
    """Add a letter marker in a black strip at the bottom right of the video"""
    def add_letter(frame):
        # Make a copy to avoid modifying the original
        frame_copy = frame.copy()
        
        # Create a black strip at the bottom of the frame
        strip_height = 60  # Height of the black strip
        strip_width = 250   # Width of the black strip
        
        # Calculate position for bottom right placement
        x_start = frame_copy.shape[1] - strip_width
        y_start = frame_copy.shape[0] - strip_height
        
        # Create black rectangle
        cv2.rectangle(frame_copy, (x_start, y_start), 
                     (frame_copy.shape[1], frame_copy.shape[0]), 
                     (0, 0, 0), -1)
        
        # Calculate text position to center in the black strip
        text_size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_DUPLEX, 2, 2)[0]
        text_x = x_start + (strip_width - text_size[0]) // 2
        text_y = y_start + (strip_height + text_size[1]) // 2
        
        # Add shadow for better visibility
        cv2.putText(frame_copy, letter, (text_x+2, text_y+2), 
                   cv2.FONT_HERSHEY_DUPLEX, 2, (50, 50, 50), 3, cv2.LINE_AA)
        
        # Add the letter in white
        cv2.putText(frame_copy, letter, (text_x, text_y), 
                   cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame_copy
    
    # Apply the letter to each frame
    return video_clip.fl_image(add_letter)


def wrap_text(text, max_width=40):
    """Wrap text to fit within a maximum width"""
    lines = []
    for paragraph in text.split('\n'):
        wrapped_lines = textwrap.wrap(paragraph, width=max_width)
        lines.extend(wrapped_lines)
        # Add an empty line between paragraphs if there's another paragraph coming
        if paragraph != text.split('\n')[-1]:
            lines.append('')
    return '\n'.join(lines)


def add_ending_question(final_clip, question_text="Which video is the most physically realistic one?\nOnly answer with a single letter (e.g., A)", fade_duration=3):
    """Add a fading question frame at the end of the video"""
    # Wrap the question text to prevent it from going off-screen
    wrapped_question = wrap_text(question_text)
    
    # Create the question frame using OpenCV
    def make_question_frame(t):
        # Create a black frame
        frame = np.zeros((final_clip.h, final_clip.w, 3), dtype=np.uint8)
        
        # Split the text into lines
        lines = wrapped_question.split('\n')
        
        # Calculate positions for centered text
        # Adjust y_position to account for possibly more lines
        y_position = final_clip.h // 2 - (len(lines) * 50) // 2
        
        # Add each line of text
        for line in lines:
            # Get text size
            text_size = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)[0]
            
            # Calculate x position to center text
            x_position = (final_clip.w - text_size[0]) // 2
            
            # Add shadow
            cv2.putText(
                frame, line, (x_position+2, y_position+2), 
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (50, 50, 50), 4, cv2.LINE_AA
            )
            
            # Add main text
            cv2.putText(
                frame, line, (x_position, y_position), 
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA
            )
            
            # Move to next line
            y_position += 60  # Reduced slightly to fit more lines
        
        # Add fade-in effect
        if t < fade_duration/2:
            alpha = t / (fade_duration/2)
            return (frame * alpha).astype(np.uint8)
        
        return frame
    
    # Create a ColorClip with the question text
    question_clip = ColorClip(
        size=(final_clip.w, final_clip.h),
        color=[0, 0, 0],
        duration=fade_duration
    )
    
    # Override the make_frame method to use our custom text function
    question_clip.make_frame = make_question_frame
    
    # Set the fps to match the main video
    question_clip = question_clip.set_fps(final_clip.fps)
    
    # Set the start time of the ending clip
    question_clip = question_clip.set_start(final_clip.duration)
    
    # Combine with the main video
    final_video = CompositeVideoClip(
        [final_clip, question_clip], 
        size=final_clip.size
    )
    
    # Set the duration to include the question frame
    final_video = final_video.set_duration(final_clip.duration + fade_duration)
    
    return final_video


def create_dynamic_video_grid(video_paths, output_path, target_width=1920, target_fps=30,
                             ending_question="Which video is the most physically realistic?\nOnly answer with a single letter (e.g., A)",
                             solutions_dir=None, questions_text_dir=None):
    """Create a video grid with flexible number of videos (supports 5 or 6 videos)"""
    # Load video clips
    original_clips = [VideoFileClip(path) for path in video_paths]
    
    # Find the minimum duration among all clips
    min_duration = min(clip.duration for clip in original_clips)
    
    # Trim all clips to the same duration
    clips = [clip.subclip(0, min_duration) for clip in original_clips]
    
    # Calculate the grid layout
    num_videos = len(clips)
    num_columns = 2  # Fixed to 2 columns
    num_rows = (num_videos + 1) // 2  # Ceiling division to get number of rows
    
    # Calculate the width for each video clip (fixed)
    clip_width = target_width // num_columns
    
    # Resize all clips while preserving aspect ratio
    resized_clips = []
    max_height = 0  # Track the tallest clip in a row
    
    for clip in clips:
        # Calculate height while preserving aspect ratio
        aspect_ratio = clip.w / clip.h
        new_height = int(clip_width / aspect_ratio)
        
        # Resize the clip
        resized_clip = clip.resize(width=clip_width)
        resized_clips.append(resized_clip)
        
        # Keep track of the maximum height in each row
        if new_height > max_height:
            max_height = new_height
    
    # Find the correct answer (video with 'full' in the name)
    correct_answer = None
    for i, path in enumerate(video_paths):
        if 'full' in Path(path).name.lower():
            correct_answer = chr(65 + i)  # Convert to letter (A, B, C, etc.)
            break
    
    # Save the correct answer to a txt file in the solutions directory
    if correct_answer and solutions_dir:
        # Extract the base name without extension from the output path
        output_filename = Path(output_path).stem
        answer_file = Path(solutions_dir) / f"{output_filename}.txt"
        with open(answer_file, 'w') as f:
            f.write(correct_answer)

        # save questioon_text
        answer_file = Path(questions_text_dir) / f"{output_filename}.txt"
        with open(answer_file, 'w') as f:
            f.write(ending_question)
        
    
    # Create letter markers (A, B, C, D, E, F, ...)
    markers = [chr(65 + i) for i in range(num_videos)]  # ASCII 65 = 'A'
    
    # Add letter markers to each clip using OpenCV
    marked_clips = [add_letter_to_video(clip, markers[i]) for i, clip in enumerate(resized_clips)]
    
    # Add letter markers to each clip
    marked_clips = [add_letter_to_video(clip, chr(65 + i)) for i, clip in enumerate(resized_clips)]  # ASCII 65 = 'A'
    
    # Calculate total height of the grid
    row_heights = []
    for r in range(num_rows):
        row_start_idx = r * num_columns
        row_end_idx = min(row_start_idx + num_columns, num_videos)
        
        # Get the maximum height of clips in this row
        if row_end_idx > row_start_idx:
            row_clips = resized_clips[row_start_idx:row_end_idx]
            row_height = max(clip.h for clip in row_clips)
            row_heights.append(row_height)
        else:
            row_heights.append(0)  # Empty row
    
    total_height = sum(row_heights)
    final_size = (target_width, total_height)
    
    # Prepare the grid arrangement
    current_y = 0
    composite_clips = []
    
    for row in range(num_rows):
        row_start_idx = row * num_columns
        row_end_idx = min(row_start_idx + num_columns, num_videos)
        row_height = row_heights[row]
        
        # Handle regular rows (full or partially filled)
        if row_end_idx - row_start_idx == num_columns:
            # Full row
            for col in range(num_columns):
                idx = row * num_columns + col
                clip = marked_clips[idx]
                composite_clips.append(clip.set_position((col * clip_width, current_y)))
        else:
            # Last row with odd number of videos - center the last video
            if num_videos % 2 == 1 and row == num_rows - 1:
                last_video = marked_clips[-1]
                center_x = (target_width - clip_width) // 2
                composite_clips.append(last_video.set_position((center_x, current_y)))
            else:
                # Partially filled row, left-aligned
                for i in range(row_start_idx, row_end_idx):
                    col = i - row_start_idx
                    clip = marked_clips[i]
                    composite_clips.append(clip.set_position((col * clip_width, current_y)))
        
        current_y += row_height
    
    # Create the final composite video
    background = ColorClip(size=final_size, color=(0, 0, 0), duration=min_duration)
    background = background.set_fps(target_fps)
    
    final_grid = CompositeVideoClip([background] + composite_clips, size=final_size)
    
    # Set the fps for the grid video
    final_grid = final_grid.set_fps(target_fps)
    
    # Add the ending question with fade
    final_clip = add_ending_question(final_grid, ending_question)
    
    # Write the result to a file
    final_clip.write_videofile(output_path, codec='libx264')
    
    # Close all clips
    for clip in original_clips + resized_clips + marked_clips:
        try:
            clip.close()
        except:
            pass
    try:
        final_grid.close()
        final_clip.close()
    except:
        pass


with open('descriptions/descriptions.csv') as f:
    reader = csv.reader(f)
    prompt_data = list(reader)

# Create output directories if they don't exist
questions_dir = Path('questions')
questions_text_dir = Path('questions_text')
solutions_dir = Path('solutions')
questions_dir.mkdir(exist_ok=True)
questions_text_dir.mkdir(exist_ok=True)
solutions_dir.mkdir(exist_ok=True)

image_root = Path("frames_selected")
images = image_root.glob("*.jpg")
images = sorted(images)

video_root = Path('videos_generated')
for image in images:
    prefix = image.stem.split('_')[0]
    video_paths = video_root.glob(f"{prefix}*.mp4")
    video_paths = [str(v) for v in sorted(video_paths)]
    # Randomize the order of videos
    import random
    random.shuffle(video_paths)

    idx = int(prefix)

    prompt = prompt_data[idx][1]
    question_templates = [  
    f"""Which of the earlier videos reflects realistic physics in this situation?\n\n{prompt}\n\nAnswer with a single letter (e.g., H). If none of them are realistic, return None.""",  
    f"""From the clips you viewed, which one accurately simulates natural motion/behavior here?\n\n{prompt}\n\nRespond using one letter (e.g., M). If none of them are realistic, return None.""",  
    f"""Which previously shown video demonstrates plausible physics for this event?\n\n{prompt}\n\nSubmit one letter (e.g., K). If none of them are realistic, return None.""",  
    f"""Which of the earlier examples exhibits scientifically valid motion/behavior?\n\n{prompt}\n\nReply with a single letter (e.g., P). If none of them are realistic, return None.""",  
    f"""Which previously shown video adheres to the laws of physics in this experiment?\n\n{prompt}\n\nAnswer with a single letter (e.g., J). If none of them are realistic, return None.""",  
    f"{prompt}\n\nWhich of the earlier videos aligns with real-world physics in this scenario?\nRespond with a single letter (e.g., K). If none of them are realistic, return None.",
    f"{prompt}\n\nFrom the clips viewed earlier, which one adheres to the laws of physics in this situation?\nChoose the correct letter (e.g., L). If none of them are realistic, return None.",
    f"{prompt}\n\nWhich demonstrated video exhibits physical plausibility for the scenario above?\nIndicate your answer as a single letter (e.g., M).",
    f"{prompt}\n\nConsidering the scene described, which of the clips shown prior follows realistic physical principles?\nProvide your answer in a single letter (e.g., Q). If none of them are realistic, return None.",
    f"{prompt}\n\nAmong the options previewed earlier, which video is consistent with real-world physics in this context?\nSubmit your answer as a single letter (e.g., R). If none of them are realistic, return None.",
    f"{prompt}\n\nWhich displayed example faithfully replicates plausible physics for the scenario above?\nAnswer with one letter only (e.g., S). If none of them are realistic, return None.",
    f"{prompt}\n\nGiven the situation described, which previously viewed video portrays motion/behavior that is physically credible?\nReply with a single letter (e.g., H). If none of them are realistic, return None.",
    f"{prompt}\n\nFrom the clips observed earlier, which one demonstrates scientifically accurate physics for the scene above?\nSelect one letter (e.g., H). If none of them are realistic, return None.",
    f"{prompt}\n\nWhich of the earlier examples aligns with the principles of physics in the context provided?\nUse a single letter to respond (e.g., J). If none of them are realistic, return None.",
    f"From the clips evaluated, which example exhibits authentic physics for the scenario outlined here?\n\n{prompt}\n\nIndicate your choice as a single letter (e.g., M). If none of them are realistic, return None.",
    f"Given the scenario, which of the earlier demonstrations maintains physical coherence effectively?\n\n{prompt}\n\nSelect one letter (e.g., N). If none of them are realistic, return None.",
    f"Which previously displayed sequence is consistent with physical laws in the following situation?\n\n{prompt}\n\nProvide your answer as a single letter (e.g., P). If none of them are realistic, return None.",
    f"From the scenarios previewed, which video aligns with observable physical interactions?\n\n{prompt}\n\nChoose a letter (e.g., S). If none of them are realistic, return None.",
    f"Which of the clips presented previously demonstrates physically authentic behavior for the scenario described?\n\n{prompt}\n\nUse a single letter to respond (e.g., T). If none of them are realistic, return None.",
    ]
    import random
    question = random.choice(question_templates)
    
    video_output_path = questions_dir / f"physics_iq_{prefix}.mp4"
    
    create_dynamic_video_grid(video_paths, str(video_output_path), target_width=1920, ending_question=question, solutions_dir=solutions_dir, questions_text_dir=questions_text_dir)
