import os
import base64
import asyncio
import subprocess
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
# pip install moviepy==1.0.3
# from moviepy.editor import VideoFileClip


def get_video_dimensions(video_path):
    """Get video dimensions using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    width, height = map(int, result.stdout.strip().split(','))
    return width, height


def resize_video_ffmpeg(video_path, max_size=512):
    """
    Resize video using FFmpeg (requires ffmpeg to be installed).
    Returns path to resized video, or None if no resize needed.
    """
    try:
        # Get video dimensions
        width, height = get_video_dimensions(video_path)
        
        # Calculate new dimensions
        if width > height:
            new_width = min(max_size, width)
            new_height = int(height * (new_width / width))
        else:
            new_height = min(max_size, height)
            new_width = int(width * (new_height / height))
        
        # Make dimensions even
        new_width = new_width if new_width % 2 == 0 else new_width - 1
        new_height = new_height if new_height % 2 == 0 else new_height - 1
        
        # Skip if already small enough
        if new_width >= width and new_height >= height:
            print(f"Video already small enough ({width}x{height}), no resize needed")
            return None
        
        print(f"Resizing video from {width}x{height} to {new_width}x{new_height}")
        
        # Create temp output path
        temp_path = f"/tmp/temp_resized_{Path(video_path).stem}_{os.getpid()}.mp4"
        
        # FFmpeg resize command
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'scale={new_width}:{new_height}',
            '-c:v', 'libx264',
            '-crf', '23',
            '-preset', 'fast',
            '-an',  # No audio
            '-y',
            temp_path
        ]
        
        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return None
        
        return temp_path
        
    except Exception as e:
        print(f"Error in resize_video_ffmpeg: {str(e)}")
        return None


def encode_b64(file_path, max_size=None):
    """
    Encode file to base64, with optional video resizing using FFmpeg.
    """
    file_ext = Path(file_path).suffix.lower()
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    # Check if it's a video and needs resizing
    if max_size and file_ext in video_extensions:
        temp_path = None
        try:
            temp_path = resize_video_ffmpeg(file_path, max_size=max_size)
            
            # Use resized video if it exists, otherwise use original
            file_to_encode = temp_path if temp_path else file_path
            
            with open(file_to_encode, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            
            return encoded
            
        except Exception as e:
            raise Exception(f"Failed to encode video: {str(e)}")
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    else:
        # Regular file encoding (no resizing)
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")


def get_video_files(folder_path, extensions=['.mp4', '.avi', '.mov', '.mkv', '.webm']):
    """Get all video files from a folder"""
    video_files = []
    folder = Path(folder_path)
    
    for ext in extensions:
        video_files.extend(folder.glob(f'*{ext}'))
        video_files.extend(folder.glob(f'*{ext.upper()}'))
    
    return sorted([str(f) for f in video_files])


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
                oldest_timestamp = self.calls_timestamps[0]
                wait_time = (oldest_timestamp + timedelta(seconds=self.period_seconds) - now).total_seconds()
                if wait_time > 0:
                    print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
            
            # Record this call
            self.calls_timestamps.append(datetime.now())


async def query_video(client, model_name, video_path, query, rate_limiter, max_size=512, max_retries=3, retry_delay=2):
    """Query a video with automatic resizing"""
    
    for attempt in range(max_retries):
        try:
            await rate_limiter.wait_if_needed()
            
            # Encode video with resizing
            try:
                base64_video = encode_b64(video_path, max_size=max_size)
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
                    retry_delay *= 2
                    continue
                return None

            # Process response
            if response is None or not hasattr(response, 'choices') or not response.choices:
                print(f"Invalid response for {video_path}")
                return None
                
            choice = response.choices[0]
            if not hasattr(choice, 'message') or not hasattr(choice.message, 'content'):
                print(f"No content in response for {video_path}")
                return ""
                
            return choice.message.content
            
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed for {video_path}: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
    
    print(f"All {max_retries} attempts failed for {video_path}")
    return None


async def process_single_video(client, model_name, video_path, output_dir, query, rate_limiter, max_size=512):
    """Process a single video and save result to text file"""
    try:
        video_name = Path(video_path).stem
        output_file = output_dir / f"{video_name}.txt"
        
        # Skip if already processed
        if output_file.exists():
            print(f"Skipping {video_name} - already processed")
            return True
        
        print(f"Processing {video_path}")
        
        # Query the model
        answer = await query_video(client, model_name, video_path, query, rate_limiter, max_size=max_size)
        
        # Save result
        if answer is not None:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(answer)
            print(f"Saved result to {output_file}")
            return True
        else:
            print(f"Failed to get response for {video_path}")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("ERROR: No response from model")
            return False
        
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        video_name = Path(video_path).stem
        output_file = output_dir / f"{video_name}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"ERROR: {str(e)}")
        return False


async def process_videos_from_folder(video_folder, model_name, query, max_concurrent=10, max_size=512, port=8000):
    """Process all videos in a folder"""
    
    # Set up client
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"
    client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    # Set up rate limiter
    rate_limiter = AsyncRateLimiter(calls_per_minute=20)
    
    # Create output directory based on model name
    model_base_name = model_name.split('/')[-1]
    output_dir = Path(model_base_name)
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get all video files
    video_files = get_video_files(video_folder)
    print(f"Found {len(video_files)} videos in {video_folder}")
    
    if not video_files:
        print("No videos found!")
        return
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(video_path):
        async with semaphore:
            return await process_single_video(
                client, model_name, video_path, output_dir, query, rate_limiter, max_size
            )
    
    # Process videos with progress bar
    tasks = [process_with_semaphore(video_path) for video_path in video_files]
    
    results = []
    for task in tqdm.as_completed(tasks, total=len(tasks), desc="Processing videos"):
        result = await task
        results.append(result)
    
    # Print summary
    successful = sum(results)
    print(f"\nCompleted: {successful}/{len(video_files)} videos processed successfully")
    
    await client.close()


##########################################################################################
################################## CONFIGURATION #########################################
##########################################################################################

# Model configuration
# model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
# model_name = "Qwen/Qwen3-VL-8B-Instruct"
# port=8002
# model_name = "Qwen/Qwen3-VL-8B-Thinking"
# port=8002
# model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
# port=8003

# model_name = "Qwen/QVQ-72B-Preview"

# Video folder path
video_folder = "../spatial_reasoning/questions"  # Change this to your video folder path

# Query to send with videos
# query = "Answer the question in this video. Generate "  # Modify as needed
query = "Answer the question in this video. Show your reasoning step-by-step, then put your final answer in \\boxed{}."

# Processing parameters
max_concurrent_queries = 5
max_video_size = 512  # Maximum dimension (width or height) in pixels

##########################################################################################
##########################################################################################
##########################################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Asynchronous Video Processing with OpenAI API")
    parser.add_argument('--video_folder', type=str, default=video_folder, help='Path to folder containing videos')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-8B-Instruct', help='Model name to use for querying')
    parser.add_argument('--port', type=int, default=8000, help='Port number for the local API server')

    model_name = parser.parse_args().model_name
    port = parser.parse_args().port
    video_folder = parser.parse_args().video_folder
    
    # Run the async processing
    asyncio.run(process_videos_from_folder(
        video_folder=video_folder,
        model_name=model_name,
        port=port,
        query=query,
        max_concurrent=max_concurrent_queries,
        max_size=max_video_size,
    ))