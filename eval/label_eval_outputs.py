import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label eval .txt outputs as correct/incorrect using ground-truth "
            "solutions from metadata.jsonl."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/train/metadata.jsonl"),
        help="Path to metadata.jsonl (default: data/train/metadata.jsonl).",
    )
    parser.add_argument(
        "--eval-dirs",
        nargs="+",
        type=Path,
        default=[Path("spatial_reasoning/eval"), Path("temporal_reasoning/eval")],
        help=(
            "One or more eval directories containing model subfolders and .txt outputs. "
            "Default: spatial_reasoning/eval temporal_reasoning/eval"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/eval_output_labels.csv"),
        help="Output CSV path (default: eval/eval_output_labels.csv).",
    )
    return parser.parse_args()


def load_solution_map(metadata_path: Path) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    solution_map: Dict[str, str] = {}
    duplicates: Dict[str, List[str]] = defaultdict(list)

    with metadata_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {metadata_path}:{line_no}: {exc}") from exc

            file_name = entry.get("file_name")
            solution = entry.get("solution")
            if not file_name or solution is None:
                continue

            stem = Path(file_name).stem
            solution = str(solution).strip()

            if stem in solution_map and solution_map[stem] != solution:
                duplicates[stem].append(solution)
            else:
                solution_map[stem] = solution

    return solution_map, duplicates


def remove_think_block(text: str) -> str:
    closing_tag = "</think>"
    if closing_tag in text:
        return text.split(closing_tag, 1)[-1].lstrip()
    return text


def extract_boxed_answers(text: str) -> List[str]:
    answers: List[str] = []
    token = r"\boxed{"
    i = 0

    while True:
        start = text.find(token, i)
        if start == -1:
            break

        j = start + len(token)
        depth = 1
        content_start = j
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1

        if depth == 0:
            answers.append(text[content_start : j - 1].strip())
            i = j
        else:
            # malformed \boxed{... without closing brace
            break

    return answers


def extract_answer(text: str) -> str:
    text = remove_think_block(text)

    boxed = extract_boxed_answers(text)
    if boxed:
        return boxed[-1]

    patterns = [
        r"(?:final answer|answer)\s*:\s*(.+?)(?:\n|$)",
        r"(?:therefore|thus|so),?\s*(?:the answer is)?\s*(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()] # get all non-blank lines
    return lines[-1] if lines else ""


def clean_latex_wrappers(text: str) -> str:
    text = text.strip().strip("$").strip()
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\left|\\right", "", text)
    return text.strip()


def normalize_text(text: str) -> str:
    text = clean_latex_wrappers(text)
    text = text.replace("−", "-")
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ",", text)
    return text


def maybe_parse_number(text: str) -> Optional[float]:
    text = text.strip()
    if not text:
        return None

    # Allow a numeric token with optional unit text around it.
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", text)
    if len(numbers) != 1:
        return None

    try:
        return float(numbers[0])
    except ValueError:
        return None


def compare_answers(prediction: str, solution: str, tol: float = 1e-5) -> bool:
    pred_norm = normalize_text(prediction)
    sol_norm = normalize_text(solution)

    if pred_norm == sol_norm:
        return True

    pred_items = [item for item in pred_norm.split(",") if item != ""]
    sol_items = [item for item in sol_norm.split(",") if item != ""]

    if len(pred_items) != len(sol_items):
        return False

    for pred_item, sol_item in zip(pred_items, sol_items):
        if pred_item == sol_item:
            continue

        pred_num = maybe_parse_number(pred_item)
        sol_num = maybe_parse_number(sol_item)

        if pred_num is None or sol_num is None:
            return False

        if abs(pred_num - sol_num) > tol:
            return False

    return True

def collect_eval_files(eval_dirs: List[Path]) -> List[Path]:
    files: List[Path] = []
    for eval_dir in eval_dirs:
        files.extend(sorted(eval_dir.rglob("*.txt")))
    return files


def get_model_name(eval_file: Path, eval_dirs: List[Path]) -> str:
    for eval_dir in eval_dirs:
        try:
            rel = eval_file.relative_to(eval_dir)
        except ValueError:
            continue
        if len(rel.parts) >= 2:
            return rel.parts[0]
    return "unknown"


def main() -> None:
    args = parse_args()

    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")

    missing_eval_dirs = [str(p) for p in args.eval_dirs if not p.exists()]
    if missing_eval_dirs:
        raise FileNotFoundError(f"Missing eval directories: {', '.join(missing_eval_dirs)}")

    solution_map, duplicate_solutions = load_solution_map(args.metadata)

    eval_files = collect_eval_files(args.eval_dirs)
    if not eval_files:
        raise FileNotFoundError("No .txt eval outputs found in the specified eval directories.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total_labeled = 0
    total_correct = 0
    missing_solution = 0
    by_model = defaultdict(lambda: {"total": 0, "correct": 0, "missing_solution": 0})

    for eval_file in eval_files:
        stem = eval_file.stem
        model_name = get_model_name(eval_file, args.eval_dirs)

        text = eval_file.read_text(encoding="utf-8", errors="replace")
        prediction = extract_answer(text)
        solution = solution_map.get(stem)

        if solution is None:
            label = "missing_solution"
            is_correct = ""
            missing_solution += 1
            by_model[model_name]["missing_solution"] += 1
        else:
            is_match = compare_answers(prediction, solution)
            label = "correct" if is_match else "incorrect"
            is_correct = str(is_match).lower()
            total_labeled += 1
            total_correct += int(is_match)
            by_model[model_name]["total"] += 1
            by_model[model_name]["correct"] += int(is_match)

        rows.append(
            {
                "eval_file": str(eval_file),
                "model_name": model_name,
                "question_stem": stem,
                "prediction_extracted": prediction,
                "solution": solution if solution is not None else "",
                "label": label,
                "is_correct": is_correct,
            }
        )

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "eval_file",
                "model_name",
                "question_stem",
                "prediction_extracted",
                "solution",
                "label",
                "is_correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote labels to: {args.output}")
    print(f"Total eval files scanned: {len(eval_files)}")
    print(f"Files with matching solution: {total_labeled}")
    print(f"Files missing solution: {missing_solution}")
    if total_labeled > 0:
        print(f"Overall accuracy: {total_correct}/{total_labeled} ({100*total_correct/total_labeled:.2f}%)")

    if duplicate_solutions:
        print(f"Warning: {len(duplicate_solutions)} duplicate stems with conflicting solutions in metadata.")

    print("\nPer-model summary:")
    for model_name in sorted(by_model):
        stats = by_model[model_name]
        total = stats["total"]
        correct = stats["correct"]
        missing = stats["missing_solution"]
        if total > 0:
            acc = 100 * correct / total
            print(f"- {model_name}: {correct}/{total} ({acc:.2f}%), missing_solution={missing}")
        else:
            print(f"- {model_name}: no matched solutions, missing_solution={missing}")


if __name__ == "__main__":
    main()
