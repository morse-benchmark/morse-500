import os
from pathlib import Path
import cv2
import numpy as np
from moviepy.editor import VideoFileClip


image_root = Path("frames_selected")
images = image_root.glob("*.jpg")
images = sorted(images)

video_root = Path('physics-IQ-benchmark/full-videos/take-1/30FPS/')
save_root = Path('videos_gt')
os.makedirs(save_root, exist_ok=True)


def find_matching_frame(video_path, template_img, threshold=0.999):
    """Find the timestamp where the template image appears in the video."""
    # Read the template image
    template = cv2.imread(str(template_img))
    if template is None:
        raise ValueError(f"Could not read template image: {template_img}")
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    best_match_val = -1
    best_match_frame = -1
    
    # Process frames
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize if needed to match dimensions
        if frame.shape[:2] != template.shape[:2]:
            frame = cv2.resize(frame, (template.shape[1], template.shape[0]))
        
        # Calculate difference between frames
        diff = cv2.absdiff(frame, template)
        diff_score = 1 - (np.sum(diff) / (diff.size * 255))
        
        # Update best match
        if diff_score > best_match_val:
            best_match_val = diff_score
            best_match_frame = frame_idx
            
        # Early stopping if we have a very good match
        if best_match_val > threshold:
            break
    
    cap.release()
    
    # If no good match found
    if best_match_val < 0.5:  # Lower threshold for reporting no match
        return None
    
    # Calculate timestamp from frame number
    timestamp = best_match_frame / fps
    return timestamp, best_match_frame, best_match_val


for image in images:
    prefix = image.stem.split('_')[0]
    video_paths = list(video_root.glob(f"{prefix}*.mp4"))
    video_paths = [str(v) for v in sorted(video_paths)]
    
    # Skip if no matching videos found
    if not video_paths:
        print(f"No videos found for prefix {prefix}")
        continue
    
    idx = int(prefix)
    print(f"Processing image {image.name} with prefix {prefix}")
    
    # Process each video file that matches the prefix
    for video_path in video_paths:
        video_name = Path(video_path).stem
        print(f"  Matching against video: {video_name}")
        
        # Find the timestamp where the frame appears in the video
        try:
            match_result = find_matching_frame(video_path, image)
            if match_result is None:
                print(f"  No matching frame found in {video_path}")
                continue
                
            timestamp, frame_num, confidence = match_result
            print(f"  Found matching frame at {timestamp:.2f}s (frame {frame_num}, confidence: {confidence:.2f})")
            
            # Load the video
            video = VideoFileClip(video_path)
            
            # Skip if timestamp is beyond video duration
            if timestamp >= video.duration:
                print(f"  Frame timestamp {timestamp}s exceeds video duration {video.duration}s")
                video.close()
                continue
            
            # Trim the video starting from the selected frame
            trimmed_video = video.subclip(timestamp)
            
            # Create output filename
            output_path = save_root / f"{video_name}_trimmed_from_{frame_num:05d}.mp4"
            
            # Save the trimmed video
            print(f"  Saving trimmed video to {output_path}")
            trimmed_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # Close the video to free resources
            video.close()
            
        except Exception as e:
            print(f"  Error processing {video_path}: {str(e)}")
