import json
import asyncio
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List


# =============================================================================
#  HELPERS (unchanged)
# =============================================================================


def extract_answer(text: str) -> str:
    """Extract the final answer from reasoning trace"""
    boxed_patterns = [
        r"\\boxed\{(.+?)\}",
        r"\$\\boxed\{(.+?)\}\$",
    ]
    for pattern in boxed_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

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


# =============================================================================
#  SUMMARY-ONLY LOGIC (single model)
# =============================================================================


async def analyze_model_predictions_summary_only(
    prediction_folder: Path,
    solutions_folder: Path,
) -> Dict:
    """
    Summary-only run:
    - Computes exact-match accuracy by comparing each prediction with its solution.
    - Aggregates category counts from existing analysis/*.json outputs, but:
        * counts a category if a file has ANY error with that category
        * skips files where is_correct == True
      (No LLM calls. No new per-question analysis is generated.)
    Returns the summary dict (also written to analysis/summary.json).
    """

    question_files = list(prediction_folder.glob("*.txt"))
    if not question_files:
        print(f"No prediction files found in: {prediction_folder}")
        return {}

    total_count = 0
    correct_count = 0

    output_folder = prediction_folder / "analysis"
    category_distribution = (
        Counter()
    )  # counts files with >=1 error in category (correct files skipped)
    subcategory_counts = Counter()  # counts per error item (correct files skipped)

    # -----------------------------
    # Accuracy over all predictions
    # -----------------------------
    for qf in question_files:
        question_name = qf.name

        with open(qf, "r", encoding="utf-8") as f:
            prediction = f.read()

        closing_tag = "</think>"
        if closing_tag in prediction:
            prediction = prediction.split(closing_tag, 1)[-1].lstrip()

        sol_path = solutions_folder / question_name
        if not sol_path.exists():
            continue

        with open(sol_path, "r", encoding="utf-8") as f:
            solution = f.read()

        total_count += 1
        if calculate_accuracy(prediction, solution):
            correct_count += 1

    # -----------------------------------------------------------------
    # Category stats from existing analysis/*.json, skipping is_correct
    # -----------------------------------------------------------------
    if output_folder.exists():
        for result_file in output_folder.glob("*.json"):
            if result_file.name == "summary.json":
                continue

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            if data.get("is_correct") is True:
                continue

            analysis = data.get("analysis_result", {}) or {}
            errors = analysis.get("errors", [])

            categories_present_in_file = set()
            if isinstance(errors, list):
                for err in errors:
                    cat = err["category"]  # err.get("category", "Unknown") or "Unknown"
                    categories_present_in_file.add(cat)

                    sub = err[
                        "subcategory"
                    ]  # err.get("subcategory", "Unknown") or "Unknown"
                    subcategory_counts[f"{cat} - {sub}"] += 1

            for cat in categories_present_in_file:
                category_distribution[cat] += 1

    acc = (correct_count / total_count) if total_count else 0.0

    # Print per-model summary
    print("\n" + "=" * 80)
    print(f"Model: {prediction_folder.parent.stem}/{prediction_folder.stem}")
    print(
        f"Accuracy (Exact Match Logic): {correct_count}/{total_count} = {acc*100:.2f}%"
    )

    if category_distribution:
        print(
            "\nCategory distribution (counting files with >=1 error in category; correct files skipped):"
        )
        denom = sum(category_distribution.values()) or 1
        for cat, count in category_distribution.most_common():
            print(f"  {cat}: {count} ({count/denom*100:.1f}%)")

    if subcategory_counts:
        print("\nTop Detailed Failures (counts per error item; correct files skipped):")
        for sub, count in subcategory_counts.most_common(10):
            print(f"  {sub}: {count}")

    # Save per-model summary.json
    output_folder.mkdir(parents=True, exist_ok=True)
    summary: Dict = {
        "model_name": prediction_folder.parent.stem,
        "task_folder": prediction_folder.stem,
        "total": total_count,
        "correct": correct_count,
        "accuracy": acc,
        "category distribution": dict(category_distribution),
        "subcategory_distribution": dict(subcategory_counts),
        "timestamp": datetime.now().isoformat(),
        "note": (
            "Summary-only run: no new per-question analysis was generated. "
            "Category distribution is computed from existing analysis/*.json, skipping is_correct==True, "
            "and counting categories by file presence (>=1 error of that category)."
        ),
    }

    with open(output_folder / "summary_new.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# =============================================================================
#  MULTI-MODEL DRIVER
# =============================================================================


async def run_all_models(
    models: List[str],
    category_name: str,
    solutions_folder: Path,
) -> None:
    all_summaries = []
    for model_name in models:
        prediction_folder = Path(f"{model_name}/{category_name}")
        summary = await analyze_model_predictions_summary_only(
            prediction_folder=prediction_folder,
            solutions_folder=solutions_folder,
        )
        if summary:
            all_summaries.append(summary)

    # Save an aggregate summary file at repo root (or adjust as you like)
    aggregate = {
        "category_name": category_name,
        "models": [s["model_name"] for s in all_summaries],
        "summaries": all_summaries,
        "timestamp": datetime.now().isoformat(),
    }
    with open(
        Path(f"summary_all_models_{category_name}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(aggregate, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Wrote aggregate summary: summary_all_models_{category_name}.json")


# =============================================================================
#  CONFIGURATION & ENTRY
# =============================================================================

MODELS = [
    # "Qwen3-VL-2B-Instruct",
    # "Qwen3-VL-2B-Thinking",
    # "Qwen3-VL-4B-Instruct",
    # "Qwen3-VL-4B-Thinking",
    # "Qwen3-VL-8B-Instruct",
    # "Qwen3-VL-8B-Thinking",
    # "Qwen3-VL-30B-A3B-Instruct-FP8",
    # "Qwen3-VL-30B-A3B-Thinking-FP8",
    # "Qwen3-VL-32B-Instruct-FP8",
    # "Qwen3-VL-32B-Thinking-FP8",
    # "Qwen3-VL-235B-A22B-Instruct-FP8",
    "Qwen3-VL-235B-A22B-Thinking-FP8",
]

CATEGORY_NAME = "spatial_reasoning"
SOLUTIONS_FOLDER = Path(f"../{CATEGORY_NAME}/solutions")

if __name__ == "__main__":
    asyncio.run(
        run_all_models(
            models=MODELS,
            category_name=CATEGORY_NAME,
            solutions_folder=SOLUTIONS_FOLDER,
        )
    )
