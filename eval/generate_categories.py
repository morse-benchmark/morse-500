import json
import asyncio
from collections import deque, Counter
from datetime import datetime, timedelta
from pathlib import Path
from openai import AsyncOpenAI
from typing import Dict, Optional, Any
import re
from tqdm.asyncio import tqdm
from string import Template

# =============================================================================
#  NEW PROMPT DEFINITION
# =============================================================================

ANALYSIS_PROMPT = Template(
    """For the following problem: $question
Carefully read the following correct reasoning trace:
$gt_reasoning

Now, examine the following reasoning trace (your task is to evaluate its accuracy step by step):                                                                                                                                                                                                                                                                                                    
$model_reasoning
                                                                                                                                                                                                                                                                                                                                                                                                    
For each step in the reasoning trace above, determine whether it is correct by comparing it to the correct reasoning trace. If a step is incorrect, briefly reflect on why it deviates. After identifying all incorrect steps, group the errors into meaningful categories based on their nature.
                                                                                                                                                                                                                                                                                                                                                                                                    
Your final output must follow this exact json format:
{
  "evaluation_summary": {
    "is_correct": boolean,
    "total_mistakes_found": integer,
    "primary_reason_for_failure": "string"
  },
  "mistakes": [
    {
      "step": "The specific string from the model reasoning that is incorrect",
      "explanation": "Why this step is wrong compared to the ground truth",
      "assigned_category": "Your custom category name"
    }, 
    ...
  ],
  "defined_categories": [
    {
      "category_name": "The custom category name used above",
      "description": "A brief definition of what this category of error represents in the context of this problem"
    }, 
    ...
  ]
}
"""
)


CATEGORIES_PROMPT = Template(
    """Condense this list of categories to a concise list of 5-10 categories, each with up to 5 subcategories. Be sure to remove each category that is too similar to another existing category:
$categories

Your output list should be in this json format: 
[
  {
    "category_name": "Name of the high-level category",
    "description": "A brief definition of what this category represents",
    "sub_categories": [
      {
        "sub_category_name": "Name of the subcategory",
        "description": "Brief description of the subcategory"
      }
    ]
  }
]
"""
)

# =============================================================================
#  HELPER CLASSES & FUNCTIONS
# =============================================================================


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
            while self.calls_timestamps and self.calls_timestamps[0] < now - timedelta(
                seconds=self.period_seconds
            ):
                self.calls_timestamps.popleft()

            if len(self.calls_timestamps) >= self.calls_per_minute:
                oldest_timestamp = self.calls_timestamps[0]
                wait_time = (
                    oldest_timestamp + timedelta(seconds=self.period_seconds) - now
                ).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

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

    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else text.strip()


def calculate_accuracy(prediction: str, ground_truth: str) -> bool:
    """Calculate if prediction matches ground truth"""
    pred_answer = extract_answer(prediction).lower().strip()
    gt_answer = ground_truth.lower().strip()

    if pred_answer == gt_answer:
        return True

    try:
        pred_num = float(re.sub(r"[^\d.-]", "", pred_answer))
        gt_num = float(re.sub(r"[^\d.-]", "", gt_answer))
        return abs(pred_num - gt_num) < 1e-5
    except:
        pass

    if gt_answer in pred_answer or pred_answer in gt_answer:
        return True

    return False


def extract_json_from_response(content: str) -> Dict:
    """Robustly extract JSON from LLM response"""
    try:
        # Try direct parse first
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to find code blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find the first '{' and last '}'
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != -1:
            json_str = content[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    return {"error": "Failed to parse JSON", "raw_content": content}


async def query_llm(
    client,
    model_name,
    query,
    rate_limiter,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    max_retries=3,
    retry_delay=2,
):

    for attempt in range(max_retries):
        try:
            await rate_limiter.wait_if_needed()

            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": query}]}
                ],
                temperature=temperature,
                # top_p=top_p, # Uncomment if your backend supports these
                # top_k=top_k,
            )

            if response and response.choices:
                return response.choices[0].message.content

        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

    return None


async def label_example(
    entry: Dict,
    rate_limiter: AsyncRateLimiter,
    client: AsyncOpenAI,
    evaluator_model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
):

    prompt = ANALYSIS_PROMPT.substitute(
        question=entry["question"],
        gt_reasoning=entry["gt_reasoning_trace"],
        model_reasoning=entry["model_output"],
    )

    return await query_llm_json(
        client,
        evaluator_model,
        prompt,
        rate_limiter,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )


async def query_llm_json(
    client,
    evaluator_model,
    query,
    rate_limiter,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
):
    content = await query_llm(
        client,
        evaluator_model,
        query,
        rate_limiter,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )

    if content is None:
        return {
            "has_error": True,
            "analysis": "LLM Request Failed",
            "primary_error_category": "API_FAILURE",
        }

    # Parse the structured JSON response
    result_json = extract_json_from_response(content)

    # If parsing failed completely, return a fallback object
    if "error" in result_json:
        return {
            "has_error": True,
            "primary_error_category": "PARSE_ERROR",
            "analysis": f"Could not parse JSON. Raw content: {content[:200]}",
        }

    return result_json


# =============================================================================
#  MAIN PROCESSING LOGIC
# =============================================================================


async def process_single_question(
    question_name: str,
    prediction_folder: Path,
    question_text_folder: Path,
    ground_truth_folder: Path,
    solutions_folder: Path,
    output_folder: Path,
    rate_limiter: AsyncRateLimiter,
    client: AsyncOpenAI,
    evaluator_model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
):
    try:
        output_file = output_folder / f"{question_name}.json"  # Save as JSON

        # Skip if already processed
        if output_file.exists():
            return True

        # Read Prediction
        pred_path = prediction_folder / question_name
        if not pred_path.exists():
            return False
        with open(pred_path, "r", encoding="utf-8") as f:
            prediction = f.read()
        # Remove anything before the final </think> tag if present
        closing_tag = "</think>"
        if closing_tag in prediction:
            # Split on the closing tag and take everything after it
            prediction = prediction.split(closing_tag, 1)[-1].lstrip()

        # Read Ground Truth (Reasoning Trace)
        gt_path = ground_truth_folder / question_name
        if not gt_path.exists():
            return False
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = f.read()

        # Read Solution (Final Answer)
        sol_path = solutions_folder / question_name
        solution = ""
        if sol_path.exists():
            with open(sol_path, "r", encoding="utf-8") as f:
                solution = f.read()

        # Read Question Text
        question_text_path = question_text_folder / question_name
        question_text = ""
        if question_text_path.exists():
            with open(question_text_path, "r", encoding="utf-8") as f:
                question_text = f.read()

        # Calculate accuracy (Simple check)
        is_correct = calculate_accuracy(prediction, solution)

        entry = {
            "question_name": question_name,
            "question": question_text,
            "solution": solution,
            "model_output": prediction,
            "gt_reasoning_trace": ground_truth,
            "is_correct": is_correct,
        }

        # ALWAYS Run Analysis now (because we have LUCKY_GUESS category for correct answers)
        # OR you can toggle this to only run if not is_correct to save money

        label_result = await label_example(
            entry,
            rate_limiter,
            client,
            evaluator_model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

        entry["analysis_result"] = label_result

        # Save result
        output_folder.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"Error processing {question_name}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


async def analyze_model_predictions(
    prediction_folder: Path,
    question_text_folder: Path,
    ground_truth_folder: Path,
    solutions_folder: Path,
    evaluator_model: str,
    max_concurrent: int,
    calls_per_minute: int,
    port: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
):

    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"
    client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base)
    rate_limiter = AsyncRateLimiter(calls_per_minute=calls_per_minute)

    output_folder = prediction_folder / "fine_analysis"
    output_folder.mkdir(exist_ok=True)

    print("prediction folder:", prediction_folder)
    question_files = list(prediction_folder.glob("*.txt"))
    if not question_files:
        print("No prediction files found!")
        return

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(question_file):
        async with semaphore:
            return await process_single_question(
                question_file.name,
                prediction_folder,
                question_text_folder,
                ground_truth_folder,
                solutions_folder,
                output_folder,
                rate_limiter,
                client,
                evaluator_model,
                temperature,
                top_p,
                top_k,
                min_p,
            )

    tasks = [process_with_semaphore(qf) for qf in question_files]
    results = []

    for task in tqdm.as_completed(tasks, total=len(tasks), desc="Analyzing"):
        result = await task
        results.append(result)

    # =========================================================================
    #  FINAL CATEGORIES GENERATION
    # =========================================================================

    total_count = 0
    correct_count = 0
    primary_category_counts = Counter()
    subcategory_counts = Counter()
    all_categories = []

    for result_file in output_folder.glob("*.json"):
        if result_file.name == "summary.json":
            continue  # skip previous summary

        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            total_count += 1
            if data.get("is_correct"):
                correct_count += 1
            analysis = data.get("analysis_result", {})
            # if type(analysis) == str:
            #     breakpoint()

            if analysis and analysis.get("defined_categories", []):
                all_categories += analysis["defined_categories"]
            else:
                print(
                    "Warning: skipped file ",
                    result_file,
                    "since it didn't have any list of categories.",
                )
    output = await query_llm_json(
        client,
        evaluator_model,
        CATEGORIES_PROMPT.substitute(categories=all_categories),
        rate_limiter,
        temperature,
        top_p,
        top_k,
        min_p,
    )

    print("\n" + "=" * 80)
    print(f"Model: {prediction_folder.stem}")
    print(
        f"Accuracy (Exact Match Logic): {correct_count}/{total_count} = {correct_count/total_count*100:.2f}%"
    )

    print(f"\nPrimary Error Categories:")
    for cat, count in primary_category_counts.most_common():
        print(f"  {cat}: {count} ({count/total_count*100:.1f}%)")

    print(f"\nTop Detailed Failures:")
    for sub, count in subcategory_counts.most_common(10):
        print(f"  {sub}: {count}")

    # Save summary
    summary = {
        "model_name": prediction_folder.stem,
        "total": total_count,
        "correct": correct_count,
        "accuracy": correct_count / total_count if total_count else 0,
        "categories": output,
    }

    with open(output_folder / "categories.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    await client.close()


# =============================================================================
#  CONFIGURATION & ENTRY
# =============================================================================

# Config
CATEGORY_NAME = "temporal_reasoning"
PREDICTION_FOLDER = Path(f"Qwen3-VL-8B-Instruct/{CATEGORY_NAME}")
QUESTION_TEXT_FOLDER = Path(f"../{CATEGORY_NAME}/question_text")
GROUND_TRUTH_FOLDER = Path(f"../{CATEGORY_NAME}/reasoning_traces")
SOLUTIONS_FOLDER = Path(f"../{CATEGORY_NAME}/solutions")
EVALUATOR_MODEL = (
    "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"  # "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
)
EVALUATOR_PORT = 8000
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MIN_P = 0.0
MAX_CONCURRENT_QUERIES = 10
CALLS_PER_MINUTE = 50

if __name__ == "__main__":
    import sys

    model_name_arg = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-VL-8B-Instruct"
    PREDICTION_FOLDER = Path(f"{model_name_arg}/{CATEGORY_NAME}")

    asyncio.run(
        analyze_model_predictions(
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
            min_p=MIN_P,
        )
    )
