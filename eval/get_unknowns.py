import json
from pathlib import Path
from typing import Any, Dict, List, Optional


MODELS = [
    "Qwen3-VL-2B-Instruct",
    "Qwen3-VL-2B-Thinking",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-4B-Thinking",
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-8B-Thinking",
    "Qwen3-VL-30B-A3B-Instruct-FP8",
    "Qwen3-VL-30B-A3B-Thinking-FP8",
    "Qwen3-VL-32B-Instruct-FP8",
    "Qwen3-VL-32B-Thinking-FP8",
    "Qwen3-VL-235B-A22B-Instruct-FP8",
]

CATEGORY_NAME = "spatial_reasoning"

# Output file written to current working directory
OUTPUT_JSON = Path(f"unknown_errors_{CATEGORY_NAME}.json")


def safe_get(d: Dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def summarize_unknown_reasons(analysis_result: Dict[str, Any]) -> List[str]:
    """
    Returns a list of reasons why this example is considered "Unknown".
    """
    reasons: List[str] = []
    primary = safe_get(analysis_result, "primary_error_category", "Unknown")

    if primary == "Unknown":
        reasons.append("primary_error_category == Unknown")

    errors = safe_get(analysis_result, "errors", [])
    if isinstance(errors, list):
        for i, err in enumerate(errors):
            cat = safe_get(err, "category", "Unknown")
            sub = safe_get(err, "subcategory", "Unknown")
            if cat == "Unknown":
                reasons.append(f"errors[{i}].category == Unknown")
            if sub == "Unknown":
                reasons.append(f"errors[{i}].subcategory == Unknown")

    return reasons


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main() -> None:
    unknown_cases: List[Dict[str, Any]] = []

    for model in MODELS:
        analysis_dir = Path(f"{model}/{CATEGORY_NAME}/analysis")
        if not analysis_dir.exists():
            print(f"[WARN] Missing analysis dir: {analysis_dir}")
            continue

        for jf in sorted(analysis_dir.glob("*.json")):
            if jf.name == "summary.json":
                continue

            data = load_json(jf)
            if not data:
                continue

            # Skip correct files
            if data.get("is_correct") is True:
                continue

            analysis_result = data.get("analysis_result", {}) or {}
            reasons = summarize_unknown_reasons(analysis_result)

            if not reasons:
                continue

            # Build a compact record; add more fields if you want
            rec = {
                "model": model,
                "question_name": data.get("question_name", jf.stem),
                "analysis_file": str(jf),
                "reasons": reasons,
                "primary_error_category": analysis_result.get(
                    "primary_error_category", None
                ),
                "unknown_error_items": [],
            }

            errors = analysis_result.get("errors", [])
            if isinstance(errors, list):
                for err in errors:
                    cat = err.get("category", "Unknown")
                    sub = err.get("subcategory", "Unknown")
                    if cat == "Unknown" or sub == "Unknown":
                        rec["unknown_error_items"].append(
                            {
                                "category": cat,
                                "subcategory": sub,
                                "severity": err.get("severity", None),
                                "evidence_quote": err.get("evidence_quote", None),
                                "description": err.get("description", None),
                            }
                        )

            unknown_cases.append(rec)

    # Print a readable summary
    print("\n" + "=" * 80)
    print(f"Unknown-error cases (skipping is_correct==True): {len(unknown_cases)}")
    print("=" * 80)

    # Show first N on console (adjust if you want)
    N = 30
    for i, rec in enumerate(unknown_cases[:N], start=1):
        print(f"\n[{i}] Model: {rec['model']}")
        print(f"    Question: {rec['question_name']}")
        print(f"    File: {rec['analysis_file']}")
        print(f"    Reasons: {', '.join(rec['reasons'])}")
        if rec["unknown_error_items"]:
            item0 = rec["unknown_error_items"][0]
            print(
                f"    Example Unknown Item: {item0.get('category')} / {item0.get('subcategory')}"
            )
            if item0.get("evidence_quote"):
                print(f"    Evidence: {item0.get('evidence_quote')}")
        else:
            print("    (No error items; only primary was Unknown)")

    # Save full report
    out = {
        "category_name": CATEGORY_NAME,
        "models_scanned": MODELS,
        "count": len(unknown_cases),
        "unknown_cases": unknown_cases,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n" + "-" * 80)
    print(f"Wrote: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
