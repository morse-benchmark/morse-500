import pandas as pd
from datasets import load_dataset
import re
from Levenshtein import distance
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_most_similar(prediction, choices):
    """Use Levenshtein distance to find the most similar choice"""
    distances = [distance(str(prediction), str(choice)) for choice in choices]
    return choices[distances.index(min(distances))]


def normalize_extracted_answer(
    extraction, choices, question_type, answer_type, precision
):
    """Normalize the extracted answer to match the answer type"""
    if question_type == "multi_choice":
        if not isinstance(extraction, str):
            try:
                extraction = str(extraction)
            except Exception:
                return None

        extraction = extraction.strip()
        if not extraction:
            return None

        sequential_chars = [chr(ord("A") + i) for i in range(len(choices))]

        if extraction in sequential_chars:
            return choices[sequential_chars.index(extraction)]
        return get_most_similar(extraction, choices)

    try:
        if answer_type == "integer":
            return str(int(float(extraction)))
        elif answer_type == "float":
            return str(round(float(extraction), int(precision)))
        elif answer_type == "list":
            return str(extraction)
    except (ValueError, TypeError):
        return None


def safe_equal(prediction, answer):
    """Safe comparison of prediction and answer"""
    try:
        return str(prediction).strip().lower() == str(answer).strip().lower()
    except Exception:
        return False


def evaluate_extractions(extracted_csv_path):
    """Evaluate extracted answers against MathVista ground truth"""
    # Load extracted results
    extracted_df = pd.read_csv(extracted_csv_path)

    # Load MathVista dataset
    mathvista = load_dataset("AI4Math/MathVista", split="testmini")

    results = []
    correct = 0

    for idx, row in tqdm(extracted_df.iterrows(), total=len(extracted_df)):
        try:
            # Get corresponding MathVista item
            mv_item = mathvista[idx]

            # Get extracted answer
            extracted = row["extracted_answer"]

            # Normalize answer based on question type
            normalized = normalize_extracted_answer(
                extraction=extracted,
                choices=mv_item.get("choices", []),
                question_type=mv_item["question_type"],
                answer_type=mv_item["answer_type"],
                precision=mv_item.get("precision", 1),
            )

            # Compare with ground truth
            is_correct = (
                safe_equal(normalized, mv_item["answer"]) if normalized else False
            )
            correct += int(is_correct)

            results.append(
                {
                    "question_id": mv_item["pid"],
                    "query": mv_item["query"],
                    "extracted": extracted,
                    "normalized": normalized,
                    "ground_truth": mv_item["answer"],
                    "is_correct": is_correct,
                    "question_type": mv_item["question_type"],
                    "answer_type": mv_item["answer_type"],
                }
            )

        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
            results.append({"question_id": idx, "error": str(e)})

    # Calculate accuracy
    accuracy = correct / len(results) if results else 0

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save evaluation results
    output_path = extracted_csv_path.replace(".csv", "_evaluated.csv")
    results_df.to_csv(output_path, index=False)

    logger.info(f"\nEvaluation complete!")
    logger.info(f"Total questions: {len(results)}")
    logger.info(f"Correct answers: {correct}")
    logger.info(f"Accuracy: {accuracy:.2%}")
    logger.info(f"Results saved to: {output_path}")

    return results_df


if __name__ == "__main__":
    # Path to your extracted answers CSV
    EXTRACTED_CSV_PATH = "mathvista_extracted_answers_image.csv"

    # Run evaluation
    evaluate_extractions(EXTRACTED_CSV_PATH)
