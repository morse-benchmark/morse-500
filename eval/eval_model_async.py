import os
import io
import time
import base64
import asyncio
from collections import deque
from datetime import datetime, timedelta
import threading
from openai import AsyncOpenAI
from datasets import load_dataset
import pandas as pd
import numpy as np
from PIL import Image
# pip install moviepy==1.0.3
from moviepy.editor import VideoFileClip
from tqdm.asyncio import tqdm


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


class AsyncRateLimiter:
    def __init__(self, calls_per_minute):
        self.calls_per_minute = calls_per_minute
        self.period_seconds = 60
        self.calls_timestamps = deque()
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        """Wait if necessary to respect the rate limit"""
        async with self.lock:
            now = datetime.now()
            
            # Remove timestamps older than the period window
            while self.calls_timestamps and self.calls_timestamps[0] < now - timedelta(seconds=self.period_seconds):
                self.calls_timestamps.popleft()
            
            # If we've reached the max calls within the window, wait until we can make another call
            if len(self.calls_timestamps) >= self.calls_per_minute:
                # Calculate how long to wait
                oldest_timestamp = self.calls_timestamps[0]
                wait_time = (oldest_timestamp + timedelta(seconds=self.period_seconds) - now).total_seconds()
                if wait_time > 0:
                    print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
            
            # Record this call
            self.calls_timestamps.append(datetime.now())


async def query_video(client, model_name, video_path, query, rate_limiter, max_retries=3, retry_delay=2):
    for attempt in range(max_retries):
        try:
            await rate_limiter.wait_if_needed()
            
            # Encode video
            try:
                base64_video = encode_b64(video_path)
                video_url = f"data:video/mp4;base64,{base64_video}"
            except Exception as e:
                print(f"Error encoding video {video_path}: {str(e)}")
                return None

            # Make API request
            try:
                response = await client.chat.completions.create(
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
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
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
            print(f"Attempt {attempt+1}/{max_retries} failed for {video_path}: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
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


async def query_video_frames(client, model_name, video_path, query, rate_limiter, fps=2, max_num_frames=32, max_retries=3, retry_delay=2):
    """
    Query a model using frames extracted from a video using moviepy.
    
    Args:
        client: AsyncOpenAI client
        model_name: The model to query
        video_path: Path to the video file
        query: Text query to send along with the frames
        rate_limiter: AsyncRateLimiter instance
        fps: Frames per second to sample (default: 2)
        max_num_frames: Maximum number of frames to use
        max_retries: Number of retries if the query fails
        retry_delay: Delay between retries in seconds
        
    Returns:
        The model's response
    """
    try:
        await rate_limiter.wait_if_needed()
        
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
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": message_content
                        }
                    ],
                )
                print(response)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error on attempt {attempt + 1}: {str(e)}. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
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


async def process_single_example(client, model_name, example, idx, video_root, rate_limiter, fps=2, max_frames=32):
    """Process a single example asynchronously"""
    try:
        video_path = f"{video_root}/" + example["video"]
        print(f"Processing {idx} {video_path}")
        query = "Answer the question in this video."
        
        # Choose between frame-based or video-based processing
        if fps is not None and max_frames is not None:
            # Use frame-based query
            print(f"Using frame-based query with fps={fps} and max_frames={max_frames}")
            answer = await query_video_frames(client, model_name, video_path, query, rate_limiter, fps=fps, max_num_frames=max_frames)
        else:
            # Use video-based query
            print("Using video-based query")
            answer = await query_video(client, model_name, video_path, query, rate_limiter)
        
        # Create result dictionary
        result = {
            "idx": idx + 1,
            "video": example["video"],
            "ground_truth": example["ground_truth"],
            "prediction": answer if answer is not None else "ERROR: No response from model",
            "question_text": example["question_text"]
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing example {idx+1}: {str(e)}")
        
        # Record the error in the results
        error_result = {
            "idx": idx + 1,
            "video": example["video"],
            "ground_truth": example.get("ground_truth", "None"),
            "prediction": f"ERROR: {str(e)}",
            "question_text": example["question_text"]
        }
        
        return error_result


async def process_dataset(dataset, model_name, size, hf_repo_path, results_file, max_concurrent=10, fps=2, max_frames=32):
    """Process the entire dataset with controlled concurrency"""
    
    # Set up client
    port = 8000
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"
    client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    # Set up rate limiter
    rate_limiter = AsyncRateLimiter(calls_per_minute=20)
    
    # Set up video root path
    video_root = f'../{hf_repo_path}/test_sz{size}'
    
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

    # Filter examples that need processing
    examples_to_process = []
    for i, example in enumerate(dataset):
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
        
        if should_process:
            examples_to_process.append((i, example))
        else:
            print(f"Skipping example {i+1} - already processed")

    print(f"Processing {len(examples_to_process)} examples with max concurrency: {max_concurrent}")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(idx_example_tuple):
        async with semaphore:
            idx, example = idx_example_tuple
            return await process_single_example(client, model_name, example, idx, video_root, rate_limiter, fps, max_frames)
    
    # Process examples with progress bar
    tasks = [process_with_semaphore(idx_example) for idx_example in examples_to_process]
    
    # Use tqdm.asyncio for async progress tracking
    results = []
    for task in tqdm.as_completed(tasks, total=len(tasks), desc="Processing videos"):
        result = await task
        results.append(result)
        
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
        
        # Sort by idx and save periodically to ensure we don't lose progress
        if len(results) % 10 == 0:  # Save every 10 results
            results_df = results_df.sort_values(by="idx").reset_index(drop=True)
            results_df.to_csv(results_file, index=False)
            print(f"Saved progress: {len(results)} results processed")

    # Final save
    results_df = results_df.sort_values(by="idx").reset_index(drop=True)
    results_df.to_csv(results_file, index=False)
    print(f"All results saved to {results_file}")
    
    await client.close()


##########################################################################################
################################## NOTES ABOUT SERVING ###################################
##########################################################################################
# For OpenAI API:
# openai_api_key = "xxx"
# model_name = "o3"
# client = AsyncOpenAI(
#     api_key=openai_api_key,
# )

# For local vLLM serving:
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
# model_name = "Qwen/QVQ-72B-Preview"

##########################################################################################
##########################################################################################
##########################################################################################

if __name__ == "__main__":
    # Load the dataset from the directory where you saved it
    hf_repo_path = f"morse-500"
    dataset = load_dataset("video-reasoning/morse-500")
    # from datasets import load_from_disk
    # dataset = load_from_disk("./morse-500-local")
    dataset = dataset['test']
    # load the 512px resized videos
    size = 512
    video_root = '../morse-500/test_sz512'

    # Load the dataset
    model_base_name = model_name.split('/')[-1]
    # video based query
    fps = max_frames = None
    results_file = f"results_sz{size}_{model_base_name}.csv"
    # images based query
    # fps = 2
    # max_frames = 32
    # results_file = f"pred_sz{size}_{model_base_name}_fps{fps}_max{max_frames}.csv"
    
    # Create a global rate limiter instance
    max_concurrent_queries = 10  # Set this to N, your desired max concurrent queries
    
    # Run the async processing
    asyncio.run(process_dataset(
        dataset=dataset,
        model_name=model_name,
        size=size,
        hf_repo_path=hf_repo_path,
        results_file=results_file,
        max_concurrent=max_concurrent_queries,
        fps=fps,
        max_frames=max_frames,
    ))