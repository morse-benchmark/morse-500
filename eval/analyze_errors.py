
import json
from pathlib import Path
from collections import deque, Counter
from datetime import datetime, timedelta
# Define the prediction folder path
# model_name = "Qwen3-VL-8B-Instruct"  # Replace with actual model name if needed
model_name = "Qwen3-VL-4B-Instruct"
# model_name = "Qwen3-VL-8B-Thinking"
# model_name = "Qwen2.5-VL-7B-Instruct"

prediction_folder = Path(f"{model_name}")
output_folder = Path(f"{model_name}_analyze")

# =========================================================================
#  SUMMARY STATISTICS GENERATION
# =========================================================================

total_count = 0
correct_count = 0
primary_category_counts = Counter()
subcategory_counts = Counter()

for result_file in output_folder.glob("*.json"):
    if result_file.name == "summary.json": continue # skip previous summary
    
    with open(result_file, 'r', encoding='utf-8') as f:
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

print("\n" + "="*80)
print(f"Model: {prediction_folder.stem}")
print(f"Accuracy (Exact Match Logic): {correct_count}/{total_count} = {correct_count/total_count*100:.2f}%")

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
    "timestamp": datetime.now().isoformat()
}
print("Summary:", summary)