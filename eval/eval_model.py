import os
import io
import time
import base64
from collections import deque
from datetime import datetime, timedelta
import threading
from openai import OpenAI
from datasets import load_dataset
import pandas as pd
import numpy as np
from PIL import Image
# pip install moviepy==1.0.3
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def get_video_stats(video_path):
    clip = VideoFileClip(video_path)
    fps = clip.fps
    duration = clip.duration
    width, height = clip.size
    clip.close()
    return fps, duration, width, height


def encode_b64(file_path):
    # encode multi-modal input to base64
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


class RateLimiter:
    def __init__(self, max_calls, period_seconds):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.calls_timestamps = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect the rate limit"""
        with self.lock:
            now = datetime.now()
            
            # Remove timestamps older than the period window
            while self.calls_timestamps and self.calls_timestamps[0] < now - timedelta(seconds=self.period_seconds):
                self.calls_timestamps.popleft()
            
            # If we've reached the max calls within the window, wait until we can make another call
            if len(self.calls_timestamps) >= self.max_calls:
                # Calculate how long to wait
                oldest_timestamp = self.calls_timestamps[0]
                wait_time = (oldest_timestamp + timedelta(seconds=self.period_seconds) - now).total_seconds()
                if wait_time > 0:
                    print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    # Ensure we're using the correct time.sleep
                    import time as time_module  # Local import to ensure we get the right module
                    time_module.sleep(wait_time)
            
            # Record this call
            self.calls_timestamps.append(datetime.now())


def query_video(model_name, video_path, query, max_retries=3, retry_delay=2):
    for attempt in range(max_retries):
        try:
            # Encode video
            try:
                base64_video = encode_b64(video_path)
                video_url = f"data:video/mp4;base64,{base64_video}"
            except Exception as e:
                print(f"Error encoding video {video_path}: {str(e)}")
                return None

            # Make API request
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": query
                                },
                                {
                                    "type": "video_url",
                                    "video_url": {"url": video_url},
                                },
                            ],
                        }
                    ],
                )
            except Exception as e:
                print(f"API request failed for {video_path}: {str(e)}")
                return None

            # Process response
            if response is None:
                print(f"Received null response for {video_path}")
                return None
                
            if not hasattr(response, 'choices') or not response.choices:
                print(f"Response for {video_path} has no choices")
                return None
                
            choice = response.choices[0]
            if not hasattr(choice, 'message') or choice.message is None:
                print(f"Choice for {video_path} has no message")
                return None
                
            if not hasattr(choice.message, 'content') or choice.message.content is None:
                print(f"Message for {video_path} has no content")
                return ""  # Return empty string instead of None to avoid subscript errors
                
            result = choice.message.content
            return result
            
        except Exception as e:
            # print(f"Unexpected error processing {video_path}: {str(e)}")
            # print(f"Error type: {type(e).__name__}")
            # return None
            print(f"Attempt {attempt+1}/{max_retries} failed for {video_path}: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
    
    print(f"All {max_retries} attempts failed for {video_path}")
    return None


def encode_image_b64(image):
    """Encode a PIL Image or numpy array to base64 string"""
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)
        
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def query_video_frames(model_name, video_path, query, fps=2, max_num_frames=32, max_retries=3, retry_delay=2):
    """
    Query a model using frames extracted from a video using moviepy.
    
    Args:
        model_name: The model to query
        video_path: Path to the video file
        query: Text query to send along with the frames
        fps: Frames per second to sample (default: 2)
        max_num_frames: Maximum number of frames to use
        max_retries: Number of retries if the query fails
        retry_delay: Delay between retries in seconds
        
    Returns:
        The model's response
    """
    try:
        rate_limiter.wait_if_needed()
        
        # Load the video
        clip = VideoFileClip(video_path)

        # Get video duration
        duration = clip.duration
        
        # First, calculate all frame timestamps at the specified fps
        frame_times = [t for t in np.arange(0, duration, 1/fps)]
        
        # If we have more frames than max_num_frames, perform uniform sampling
        if len(frame_times) > max_num_frames:
            # Calculate indices for uniform sampling
            indices = np.linspace(0, len(frame_times) - 1, max_num_frames, dtype=int)
            frame_times = [frame_times[i] for i in indices]
        
        # Extract frames at calculated timestamps
        frames = []
        for t in frame_times:
            # Get frame at time t
            frame = clip.get_frame(min(t, duration))  # Ensure we don't exceed duration
            # Convert to PIL Image
            pil_image = Image.fromarray(frame)
            frames.append(pil_image)
            
        # Close the clip to free resources
        clip.close()
        
        print(f"Extracted {len(frames)} frames from video (duration: {duration:.2f}s)")
        
        # Create the message content with multiple images
        message_content = [{"type": "text", "text": query}]
        
        # Add each frame as an image
        for frame in frames:
            base64_frame = encode_image_b64(frame)
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"}
            })
        
        # Query the model with retries
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": message_content
                        }
                    ],
                )
                print(response)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error on attempt {attempt + 1}: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed after {max_retries} attempts: {str(e)}")
                    return None
            
            # Process response
            if response is None:
                print(f"Received null response for {video_path}")
                return None
                
            if not hasattr(response, 'choices') or not response.choices:
                print(f"Response for {video_path} has no choices")
                return None
                
            choice = response.choices[0]
            if not hasattr(choice, 'message') or choice.message is None:
                print(f"Choice for {video_path} has no message")
                return None
                
            if not hasattr(choice.message, 'content') or choice.message.content is None:
                print(f"Message for {video_path} has no content")
                return ""  # Return empty string instead of None to avoid subscript errors
                
            result = choice.message.content
            return result

    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None
    

##########################################################################################
################################## NOTES ABOUT SERVING ###################################
##########################################################################################
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "xxx"
model_name = "o3"
client = OpenAI(
    api_key=openai_api_key,
)
##########################################################################################
##########################################################################################
##########################################################################################
# Add rate limiting if needed
# Create a global rate limiter instance (20 calls per minute)
rate_limiter = RateLimiter(max_calls=20, period_seconds=60)


if __name__ == "__main__":
    
    # Load the dataset from the directory where you saved it
    dataset = load_dataset("video-reasoning/morse-500")
    dataset = dataset['test']
    # load the 512px resized videos
    size = 512
    video_root = 'morse-500/test_sz512'
    
    model_base_name = model_name.split('/')[-1]
    # image based model
    fps = 2
    max_frames = 32
    results_file = f"pred_sz{size}_{model_base_name}_fps{fps}_max{max_frames}.csv"
    # video based model
    # results_file = f"pred_sz{size}_{model_base_name}.csv"

    # Initialize results DataFrame - either from existing file or new
    if os.path.exists(results_file):
        print(f"Loading existing results from {results_file}")
        results_df = pd.read_csv(results_file)
        
        # Create a dictionary mapping idx to prediction for quick lookup
        existing_results = {}
        for _, row in results_df.iterrows():
            existing_results[row['idx']] = row.get('prediction', '')
        
        print(f"Loaded {len(results_df)} existing results")
    else:
        print(f"No existing results file found. Creating new results.")
        results_df = pd.DataFrame(columns=["idx", "video", "ground_truth", "prediction", "question_text"])
        existing_results = {}

    # Process examples one by one
    for i, example in tqdm(enumerate(dataset)):
        # Skip examples that already have non-empty predictions
        example_idx = i + 1  # Match the idx in results
        
        # Check if we should process this example
        prediction = existing_results.get(example_idx, None)
        should_process = (
            example_idx not in existing_results or 
            prediction is None or
            pd.isna(prediction) or
            prediction == '' or
            (isinstance(prediction, str) and prediction.startswith("ERROR"))
        )
        
        if not should_process:
            print(f"Skipping example {i+1} - already processed")
            continue
            
        # Process this example
        try:
            video_path = f"{video_root}/" + example["video"]
            print(f"Processing {i} {video_path}")
            query = "Answer the question in this video."
            answer = query_video_frames(model_name, video_path, query, fps=fps, max_num_frames=max_frames)
            # answer = query_video(model_name, video_path, query)
            
            # Create result dictionary
            result = {
                "idx": i+1,
                "video": example["video"],
                "ground_truth": example["ground_truth"],
                "prediction": answer if answer is not None else "ERROR: No response from model",
                "question_text": example["question_text"]
            }
            
            # Update the DataFrame with the new result
            idx_value = result["idx"]
            
            # If row exists, update it; otherwise, append it
            if idx_value in existing_results:
                # Update existing row
                mask = results_df['idx'] == idx_value
                for col, value in result.items():
                    results_df.loc[mask, col] = value
            else:
                # Append new row
                results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
            
            # Update existing_results dictionary
            existing_results[idx_value] = result["prediction"]
            
            # Print progress
            print(f"Processed example {i+1}/{len(dataset)}")
            
            # Sort by idx and save after error to ensure we don't lose progress
            results_df = results_df.sort_values(by="idx").reset_index(drop=True)
            # Save after each example to ensure we don't lose progress
            results_df.to_csv(results_file, index=False)
            
        except Exception as e:
            print(f"Error processing example {i+1}: {str(e)}")
            
            # Record the error in the results
            error_result = {
                "idx": i+1,
                "video": example["video"],
                "ground_truth": example.get("ground_truth", "None"),
                "prediction": f"ERROR: {str(e)}",
                "question_text": example["question_text"]
            }
            
            # Update the DataFrame with the error result
            if i+1 in existing_results:
                mask = results_df['idx'] == i+1
                for col, value in error_result.items():
                    results_df.loc[mask, col] = value
            else:
                results_df = pd.concat([results_df, pd.DataFrame([error_result])], ignore_index=True)
            
            # Update existing_results dictionary
            existing_results[i+1] = error_result["prediction"]
            
            # Sort by idx and save after error to ensure we don't lose progress
            results_df = results_df.sort_values(by="idx").reset_index(drop=True)
            # Save after error to ensure we don't lose progress
            results_df.to_csv(results_file, index=False)

    # Final save
    # Sort by idx and save after error to ensure we don't lose progress
    results_df = results_df.sort_values(by="idx").reset_index(drop=True)
    results_df.to_csv(results_file, index=False)
    print(f"All results saved to {results_file}")
