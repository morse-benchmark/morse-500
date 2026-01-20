import json
import asyncio
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from openai import AsyncOpenAI
from typing import Dict, Optional
import re
from tqdm.asyncio import tqdm


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
                    # print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
            
            # Record this call
            self.calls_timestamps.append(datetime.now())

def extract_answer(text: str) -> str:
    """Extract the final answer from reasoning trace"""
    # Look for boxed answer first
    boxed_patterns = [
        r"\\boxed\{(.+?)\}",
        r"\$\\boxed\{(.+?)\}\$",
    ]
    
    for pattern in boxed_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Look for common answer patterns
    patterns = [
        r"(?:final answer|answer):\s*(.+?)(?:\n|$)",
        r"(?:therefore|thus|so),?\s*(?:the answer is)?\s*(.+?)(?:\n|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no pattern matches, return the last non-empty line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else text.strip()

def calculate_accuracy(prediction: str, ground_truth: str) -> bool:
    """Calculate if prediction matches ground truth"""
    pred_answer = extract_answer(prediction).lower().strip()
    gt_answer = ground_truth.lower().strip()
    
    # Exact match
    if pred_answer == gt_answer:
        return True
    
    # Numeric comparison
    try:
        pred_num = float(re.sub(r'[^\d.-]', '', pred_answer))
        gt_num = float(re.sub(r'[^\d.-]', '', gt_answer))
        return abs(pred_num - gt_num) < 1e-5
    except:
        pass
    
    # Substring match (for verbose answers)
    if gt_answer in pred_answer or pred_answer in gt_answer:
        return True
    
    return False

async def query_llm(client, model_name, query, rate_limiter, 
                    temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
                    max_retries=3, retry_delay=2):
    """Query LLM with text input only"""
    
    for attempt in range(max_retries):
        try:
            await rate_limiter.wait_if_needed()

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
                                }
                            ],
                        }
                    ],
                    temperature=temperature,
                )
            except Exception as e:
                print(f"API request failed: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return None

            # Process response
            if response is None or not hasattr(response, 'choices') or not response.choices:
                print(f"Invalid response")
                return None
                
            choice = response.choices[0]
            if not hasattr(choice, 'message') or not hasattr(choice.message, 'content'):
                print(f"No content in response")
                return ""
                
            return choice.message.content
            
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
    
    print(f"All {max_retries} attempts failed")
    return None

async def label_example(entry: Dict, rate_limiter: AsyncRateLimiter, client: AsyncOpenAI, 
                       evaluator_model: str = "gpt-4o",
                       temperature: float = 0.7, top_p: float = 0.8, 
                       top_k: int = 20, min_p: float = 0.0):
    """Label a single example with rate limiting"""
    
    prompt = f"""Analyze this model failure and identify ALL applicable failure modes from the taxonomy below.

TAXONOMY:
1. Perception Errors - misidentifies objects, attributes, spatial layout, or visual relations
2. Interpretation Errors - perceives correctly but misinterprets meaning or context  
3. Reasoning Errors - logical, mathematical, or deductive mistakes
4. Instruction Errors - ignores or misreads explicit instructions
5. Hallucination / Fabrication - introduces information not present
6. Other - specify what type of error

USER QUESTION:
{entry['question']}

MODEL OUTPUT:
{entry['model_output']}

GROUND TRUTH:
{entry['gt_reasoning_trace']}

List ALL failure modes that apply (there may be multiple). For each one, explain in 1-2 sentences WHY it applies.
Note: even if the final answer is correct, the intermediate steps might be wrong. Examine carefully.

Format (one failure mode per line):
- [Failure Mode Name]: [explanation]
- [Failure Mode Name]: [explanation]
...

Example:
- Reasoning Errors: The model correctly identified the objects but made an arithmetic mistake when calculating the total distance.
- Interpretation Errors: The model misunderstood that "left of" means from the viewer's perspective, not the object's perspective.
"""
    
    content = await query_llm(client, evaluator_model, prompt, rate_limiter,
                             temperature=temperature, top_p=top_p, 
                             top_k=top_k, min_p=min_p)
    
    if content is None:
        return {"labels": [], "explanations": {}, "raw_response": "Failed to get response"}
    
    # Parse the response
    labels = []
    explanations = {}
    
    try:
        lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
        
        for line in lines:
            # Look for lines starting with "-" or bullet points
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                line = line[1:].strip()  # Remove bullet
                
                # Split on first colon
                if ':' in line:
                    parts = line.split(':', 1)
                    label = parts[0].strip()
                    explanation = parts[1].strip()
                    
                    # Normalize label name
                    label = normalize_label(label)
                    
                    labels.append(label)
                    explanations[label] = explanation
        
        # If no labels found, try without bullet points
        if not labels:
            for line in lines:
                if ':' in line:
                    parts = line.split(':', 1)
                    label = parts[0].strip()
                    explanation = parts[1].strip()
                    
                    # Check if this looks like a taxonomy label
                    label_lower = label.lower()
                    if any(keyword in label_lower for keyword in ['perception', 'interpretation', 'reasoning', 'instruction', 'hallucination', 'fabrication', 'error', 'other']):
                        label = normalize_label(label)
                        labels.append(label)
                        explanations[label] = explanation
        
    except Exception as e:
        print(f"Parse error: {e}")
        print(f"Content: {content[:500]}")
        return {
            "labels": ["Parse Error"],
            "explanations": {"Parse Error": content[:200] if content else str(e)},
            "raw_response": content
        }
    
    # If still no labels found, return the raw response
    if not labels:
        return {
            "labels": ["Unknown"],
            "explanations": {"Unknown": content[:300]},
            "raw_response": content
        }
    
    return {
        "labels": labels,
        "explanations": explanations,
        "raw_response": content
    }


def normalize_label(label: str) -> str:
    """Normalize label names to standard taxonomy"""
    label = label.strip()
    
    # Remove numbers and clean up
    label = re.sub(r'^\d+\.\s*', '', label)
    label = label.replace('**', '').strip()
    
    # Map to standard taxonomy
    label_lower = label.lower()
    
    if 'perception' in label_lower:
        return "Perception Errors"
    elif 'interpretation' in label_lower:
        return "Interpretation Errors"
    elif 'reasoning' in label_lower:
        return "Reasoning Errors"
    elif 'instruction' in label_lower:
        return "Instruction Errors"
    elif 'hallucination' in label_lower or 'fabrication' in label_lower:
        return "Hallucination / Fabrication"
    elif 'other' in label_lower:
        return "Other"
    else:
        # Return as-is if doesn't match known categories
        return label.title()


async def process_single_question(
    question_name: str,
    prediction_folder: Path,
    question_text_folder: Path,
    ground_truth_folder: Path,
    solutions_folder: Path,
    output_folder: Path,
    rate_limiter: AsyncRateLimiter,
    client: AsyncOpenAI,
    evaluator_model: str = "gpt-4o",
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0
):
    """Process a single question"""
    try:
        output_file = output_folder / question_name
        
        # Skip if already processed
        if output_file.exists():
            # print(f"Skipping {question_name} - already processed")
            return True
        
        # Read prediction
        pred_path = prediction_folder / question_name
        if not pred_path.exists():
            print(f"Prediction not found: {pred_path}")
            return False
        
        with open(pred_path, 'r', encoding='utf-8') as f:
            prediction = f.read()
        
        # Read ground truth
        gt_path = ground_truth_folder / question_name
        if not gt_path.exists():
            print(f"Ground truth not found: {gt_path}")
            return False
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            ground_truth = f.read()
        
        # Read solution 
        sol_path = solutions_folder / question_name
        if sol_path.exists():
            with open(sol_path, 'r', encoding='utf-8') as f:
                solution = f.read()
        
        question_text_path = question_text_folder / question_name
        if question_text_path.exists():
            with open(question_text_path, 'r', encoding='utf-8') as f:
                question_text = f.read()


        # Calculate accuracy
        is_correct = calculate_accuracy(prediction, solution)
        
        # Prepare entry for labeling
        entry = {
            "question_name": question_name,
            "question": question_text,
            "solution": solution,
            "model_output": prediction,
            "gt_reasoning_trace": ground_truth,
            "is_correct": is_correct,
            "model_name": str(prediction_folder)
        }
        
        # Label the example (only if incorrect to save API calls)
        label_result = await label_example(
            entry, rate_limiter, client, evaluator_model,
            temperature=temperature, top_p=top_p, 
            top_k=top_k, min_p=min_p
        )
        entry["failure_analysis"] = label_result
        
        # Save result
        output_folder.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
        
        # print(f"✓ {question_name} - Correct: {is_correct} - Label: {entry['label_result'].get('label', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"Error processing {question_name}: {str(e)}")
        return False

async def analyze_model_predictions(
    prediction_folder: Path,
    question_text_folder: Path,
    ground_truth_folder: Path,
    solutions_folder: Path,
    evaluator_model: str = "gpt-4o",
    max_concurrent: int = 10,
    calls_per_minute: int = 30,
    port: int = 8000,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0
):
    """Analyze all predictions from a model folder"""
    
    # Set up OpenAI client
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"
    client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    # Set up rate limiter
    rate_limiter = AsyncRateLimiter(calls_per_minute=calls_per_minute)
    
    # Create output directory
    output_folder = Path(f"{prediction_folder.stem}_analyze")
    output_folder.mkdir(exist_ok=True)
    print(f"Output directory: {output_folder}")
    
    # Get all prediction files
    if not prediction_folder.exists():
        print(f"Prediction folder not found: {prediction_folder}")
        return
    
    question_files = list(prediction_folder.glob("*.txt"))
    print(f"Found {len(question_files)} questions to analyze")
    
    if not question_files:
        print("No prediction files found!")
        return
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(question_file):
        async with semaphore:
            return await process_single_question(
                question_name=question_file.name,
                prediction_folder=prediction_folder,
                question_text_folder=question_text_folder,
                ground_truth_folder=ground_truth_folder,
                solutions_folder=solutions_folder,
                output_folder=output_folder,
                rate_limiter=rate_limiter,
                client=client,
                evaluator_model=evaluator_model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p
            )
    
    # Process all questions with progress bar
    tasks = [process_with_semaphore(qf) for qf in question_files]
    
    results = []
    for task in tqdm.as_completed(tasks, total=len(tasks), desc="Analyzing predictions"):
        result = await task
        results.append(result)
    
    # Calculate overall statistics
    correct_count = 0
    total_count = 0
    label_counts = {}
    
    for result_file in output_folder.glob("*.txt"):
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
            total_count += 1
            if result.get("is_correct"):
                correct_count += 1
            
            # Count all labels (not just primary)
            failure_analysis = result.get("failure_analysis", {})
            labels = failure_analysis.get("labels", [])
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\n" + "="*80)
    print(f"Model: {prediction_folder.stem}")
    print(f"Overall Accuracy: {correct_count}/{total_count} = {correct_count/total_count*100:.2f}%")
    print(f"\nFailure Mode Distribution (may sum to more than total errors due to multiple labels):")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count} ({count/total_count*100:.1f}%)")
    
    # Save summary statistics
    summary = {
        "model_name": prediction_folder.stem,
        "total_questions": total_count,
        "correct": correct_count,
        "accuracy": correct_count / total_count if total_count > 0 else 0,
        "error_distribution": label_counts,
        "sampling_params": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p
        },
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    with open(output_folder / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary saved to {output_folder / 'summary.json'}")
    print("="*80)
    
    await client.close()


##########################################################################################
################################## CONFIGURATION #########################################
##########################################################################################

# Model prediction folder to analyze
PREDICTION_FOLDER = Path("Qwen3-VL-8B-Instruct")  # Change this to your model output folder

# Reference folders
QUESTION_TEXT_FOLDER = Path("../spatial_reasoning/question_text")
GROUND_TRUTH_FOLDER = Path("../spatial_reasoning/reasoning_traces")  # Folder with ground truth reasoning traces
SOLUTIONS_FOLDER = Path("../spatial_reasoning/solutions")  # Folder with solution files

# Evaluator model configuration (OpenAI-compatible API)
# EVALUATOR_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
# EVALUATOR_MODEL = "Qwen/Qwen3-32B"
EVALUATOR_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
EVALUATOR_PORT = 8000  # Port for the evaluator model API

# Sampling parameters for Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MIN_P = 0.0

# Processing parameters
MAX_CONCURRENT_QUERIES = 5
CALLS_PER_MINUTE = 30

##########################################################################################
##########################################################################################
##########################################################################################

if __name__ == "__main__":
    # Run the async analysis

    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-VL-8B-Instruct"
    PREDICTION_FOLDER = Path(model_name)

    asyncio.run(analyze_model_predictions(
        prediction_folder=PREDICTION_FOLDER,
        question_text_folder=QUESTION_TEXT_FOLDER,
        ground_truth_folder=GROUND_TRUTH_FOLDER,
        solutions_folder=SOLUTIONS_FOLDER,
        evaluator_model=EVALUATOR_MODEL,
        max_concurrent=MAX_CONCURRENT_QUERIES,
        calls_per_minute=CALLS_PER_MINUTE,
        port=EVALUATOR_PORT,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        min_p=MIN_P
    ))