import os
import re
import io
from collections import deque
from datetime import datetime, timedelta
import time
import pandas as pd
from PIL import Image

from google import genai
from google.genai import types
google_api_key = "xxx"
model_name = "veo-2.0-generate-001"
client = genai.Client(api_key=google_api_key)


class RateLimiter:
    """Rate limiter to control API request frequency"""
    
    def __init__(self, max_requests=2, time_window_minutes=1):
        self.max_requests = max_requests
        self.time_window = timedelta(minutes=time_window_minutes)
        self.requests = deque()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        now = datetime.now()
        
        # Remove requests older than the time window
        while self.requests and now - self.requests[0] > self.time_window:
            self.requests.popleft()
        
        # If we're at the limit, wait until we can make another request
        if len(self.requests) >= self.max_requests:
            sleep_time = (self.requests[0] + self.time_window - now).total_seconds()
            if sleep_time > 0:
                print(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                # Clean up old requests after waiting
                now = datetime.now()
                while self.requests and now - self.requests[0] > self.time_window:
                    self.requests.popleft()
        
        # Record this request
        self.requests.append(now)
        print(f"API calls in last minute: {len(self.requests)}")

def extract_scenario_id(filename):
    """Extract scenario ID from filename (e.g., '0013' from '0013_switch-frames_...')"""
    match = re.match(r'(\d{4})', filename)
    return match.group(1) if match else None

def find_matching_image(scenario_id, frames_folder):
    """Find the image file that matches the scenario ID"""
    for filename in os.listdir(frames_folder):
        if filename.startswith(scenario_id) and filename.endswith('.jpg'):
            return os.path.join(frames_folder, filename)
    return None

def generate_single_video(image_path, prompt, output_name, client):
    """
    Generate a single video from an image and prompt
    
    Args:
        image_path: Path to the input image
        prompt: Text description for video generation
        output_name: Name for the output video file
        client: Google AI Platform client
    """
    try:
        # Apply rate limiting if provided
        if rate_limiter:
            rate_limiter.wait_if_needed()
        
        im = Image.open(image_path)
        # converting the image to bytes
        image_bytes_io = io.BytesIO()
        im.save(image_bytes_io, format=im.format)
        image_bytes = image_bytes_io.getvalue()
        image=types.Image(image_bytes=image_bytes, mime_type=im.format)

        # Generate video
        operation = client.models.generate_videos(
            model=model_name,
            prompt=prompt,
            image=image,
            config=types.GenerateVideosConfig(
                person_generation="dont_allow",
                aspect_ratio="16:9",
                number_of_videos=1,
                duration_seconds=5
            ),
        )
        
        # Wait for completion
        while not operation.done:
            time.sleep(20)
            operation = client.operations.get(operation)
            print(operation)
        
        # Save video
        if operation.response.generated_videos:
            video = operation.response.generated_videos[0]
            client.files.download(file=video.video)
            video.video.save(output_name)
            print(f"Video saved as: {output_name}")
        else:
            print("No video generated")
            
    except Exception as e:
        print(f"Error generating video: {str(e)}")



rate_limiter = RateLimiter(max_requests=2, time_window_minutes=1)
CSV_PATH = "descriptions/descriptions.csv"
FRAMES_FOLDER = "frames_selected"
OUTPUT_FOLDER = "videos_generated"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Read CSV data
df = pd.read_csv(CSV_PATH)

# Process each row in the CSV
for index, row in df.iterrows():
    scenario = row['scenario']
    description = row['description']
    category = row['category']
    generated_video_name = row['generated_video_name']
    
    print(f"\nProcessing scenario: {scenario}")
    print(f"Description: {description[:100]}...")
    
    # Extract scenario ID from the scenario column
    scenario_id = extract_scenario_id(scenario)
    if not scenario_id:
        print(f"Could not extract scenario ID from: {scenario}")
        continue
        
    # Find matching image
    image_path = find_matching_image(scenario_id, FRAMES_FOLDER)
    if not image_path:
        print(f"No matching image found for scenario ID: {scenario_id}")
        continue
        
    print(f"Using image: {os.path.basename(image_path)}")
    
    prompt = description
    scenario_idx = scenario.split('_')[0]
    output_name = f"{OUTPUT_FOLDER}/{scenario_idx}_veo2.mp4"
    if int(scenario_idx) <= 50: 
        continue
    generate_single_video(image_path, prompt, output_name, client)

