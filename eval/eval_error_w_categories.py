import json
import asyncio
from collections import deque, Counter
from datetime import datetime, timedelta
from pathlib import Path
from openai import AsyncOpenAI
from typing import Dict, Optional, Any
import re
from tqdm.asyncio import tqdm

# =============================================================================
#  NEW PROMPT DEFINITION
# =============================================================================

ANALYSIS_PROMPT = """You are an expert evaluator analyzing vision-language model errors.

Given:
- Question: {question}
- Ground Truth Reasoning: {gt_reasoning}
- Ground Truth Answer: {gt_answer}
- Model Reasoning: {model_reasoning}
- Model Answer: {model_answer}

ERROR CATEGORIES:

{categories}

OUTPUT FORMAT (JSON):
{{
  "has_error": true/false,
  "primary_error_category": [category name from list of categories],
  "errors": [
    {{
      "category": [category name from list of categories],
      "subcategory": [subcategory name from list of subcategories under assigned category],
      "evidence_quote": [quoted line(s) from model reasoning],
      "description": [explanation of how it's wrong according to the ground truth reasoning],
      "severity": [if it's a major or minor mistake]
    }}, 
    ...
  ],
  "analysis": [concise summary of errors]
}}

Now analyze:
"""

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
    c = "".join(
        [
            f"{i + 1}. {category['category_name']}"
            + "".join(
                [
                    f"  - {subcategory['sub_category_name']}: {subcategory['description']}\n"
                    for subcategory in category["sub_categories"]
                ]
            )
            for i, category in enumerate(entry["categories"])
        ]
    )
    # Fill the prompt template
    prompt = ANALYSIS_PROMPT.format(
        question=entry["question"],
        gt_reasoning=entry["gt_reasoning_trace"],
        gt_answer=entry["solution"],
        model_reasoning=entry["model_output"],
        model_answer=extract_answer(entry["model_output"]),
        categories=c,
    )

    content = await query_llm(
        client,
        evaluator_model,
        prompt,
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
    categories_file: Path,
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

        # Read Categories Text
        if categories_file.exists():
            with open(categories_file, "r", encoding="utf-8") as f:
                categories_data = json.load(f)
                categories_list = categories_data["categories"]

        # Calculate accuracy (Simple check)
        is_correct = calculate_accuracy(prediction, solution)

        entry = {
            "question_name": question_name,
            "question": question_text,
            "solution": solution,
            "model_output": prediction,
            "gt_reasoning_trace": ground_truth,
            "is_correct": is_correct,
            "categories": categories_list,
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
    categories_file: Path,
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

    output_folder = Path(f"{prediction_folder.stem}_analyze")
    output_folder.mkdir(exist_ok=True)

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
                categories_file,
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
    #  SUMMARY STATISTICS GENERATION
    # =========================================================================

    total_count = 0
    correct_count = 0
    primary_category_counts = Counter()
    subcategory_counts = Counter()

    for result_file in output_folder.glob("*.json"):
        if result_file.name == "summary.json":
            continue  # skip previous summary

        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            total_count += 1
            if data.get("is_correct"):
                correct_count += 1

            analysis = data.get("analysis_result", {})

            # Count Primary Category
            primary = analysis.get("primary_error_category", "Unknown")
            if primary:
                primary_category_counts[primary] += 1

            # Count Subcategories
            errors = analysis.get("errors", [])
            if isinstance(errors, list):
                for err in errors:
                    sub = err.get("subcategory", "Unknown")
                    cat = err.get("category", "Unknown")
                    subcategory_counts[f"{cat} - {sub}"] += 1

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
        "primary_distribution": dict(primary_category_counts),
        "subcategory_distribution": dict(subcategory_counts),
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    await client.close()


# =============================================================================
#  CONFIGURATION & ENTRY
# =============================================================================

# Config
PREDICTION_FOLDER = Path("Qwen3-VL-8B-Instruct")
CATEGORIES_FILE = Path("Qwen3-VL-8B-Instruct_analyze/simplified_categories.json")
QUESTION_TEXT_FOLDER = Path("../spatial_reasoning/question_text")
GROUND_TRUTH_FOLDER = Path("../spatial_reasoning/reasoning_traces")
SOLUTIONS_FOLDER = Path("../spatial_reasoning/solutions")
EVALUATOR_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
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
    PREDICTION_FOLDER = Path(model_name_arg)
    CATEGORIES_FILE = Path(f"{model_name_arg}_analyze/simplified_categories.json")

    asyncio.run(
        analyze_model_predictions(
            prediction_folder=PREDICTION_FOLDER,
            categories_file=CATEGORIES_FILE,
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
