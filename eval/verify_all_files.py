import os
from pathlib import Path

# SET THIS TO False TO ACTUALLY DELETE FILES
DRY_RUN = True

# List of specific models to process
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
    "Qwen3-VL-235B-A22B-Thinking-FP8",
]


def get_basenames(directory):
    """Returns a dictionary mapping the basename (stripping .txt.json) to the full filename."""
    if not os.path.exists(directory):
        return {}

    mapping = {}
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            # Handle .txt.json specifically to get the pure basename
            if f.endswith(".txt.json"):
                basename = f[:-9]  # Strips '.txt.json'
            else:
                basename = os.path.splitext(f)[0]

            mapping[basename] = f
    return mapping


def cleanup_by_basenames(base_path="."):
    # 1. Map Global Reference Basenames
    ref_paths = {
        "spatial": Path("/usr/workspace/cai6/morse-500/spatial_reasoning/questions"),
        "temporal": Path("/usr/workspace/cai6/morse-500/temporal_reasoning/questions"),
    }

    references = {}
    for key, path in ref_paths.items():
        if path.exists():
            references[key] = get_basenames(path)
            print(
                f"Loaded {len(references[key])} reference basenames for {key.upper()}."
            )
        else:
            print(f"Error: Global path {path} not found.")
            return

    print(f"Checking {len(MODELS)} specified model directories.\n")

    for model in MODELS:
        model_full_path = Path(base_path) / model
        if not model_full_path.exists():
            print(f"=== Model: {model} [SKIPPED - Directory not found] ===")
            continue

        print(f"=== Model: {model} ===")

        for category in ["spatial", "temporal"]:
            root_dir = model_full_path / f"{category}_reasoning"
            analysis_dir = root_dir / "analysis"

            # --- 1. PROCESS ROOT FOLDER (Questions) ---
            if root_dir.exists():
                print(f"  > Checking Root: {root_dir.relative_to(base_path)}")

                current_map = get_basenames(root_dir)
                ref_basenames = set(references[category].keys())
                current_basenames = set(current_map.keys())

                # Root: Ignore nothing; extra files (including stray summaries) are flagged
                extra = current_basenames - ref_basenames
                missing = ref_basenames - current_basenames

                if missing:
                    print(f"    [✘] {len(missing)} files missing.")

                if extra:
                    print(f"    [!] {len(extra)} extra files found.")
                    for e in extra:
                        full_filename = current_map[e]
                        file_to_delete = root_dir / full_filename
                        if DRY_RUN:
                            print(f"        [DRY RUN] Would delete: {full_filename}")
                        else:
                            try:
                                os.remove(file_to_delete)
                                print(f"        [DELETED] {full_filename}")
                            except OSError as err:
                                print(f"        [ERROR] {full_filename}: {err}")

                if not missing and not extra:
                    print(f"    [✓] Root synced.")
            else:
                print(f"  [!] {category.upper()} Root folder missing at {root_dir}")

            # --- 2. PROCESS ANALYSIS FOLDER (Summary + Cleanup) ---
            if analysis_dir.exists():
                print(f"  > Checking Analysis: {analysis_dir.relative_to(base_path)}")

                # Check for summary_new.json specifically here
                summary_file = analysis_dir / "summary_new.json"
                if summary_file.exists():
                    print(f"    [✓] 'summary_new.json' found.")
                else:
                    print(f"    [⚠] 'summary_new.json' is MISSING.")

                current_map = get_basenames(analysis_dir)
                current_basenames = set(current_map.keys())

                # Ignore 'summary_new' from deletion logic in analysis
                extra = (current_basenames - ref_basenames) - {"summary_new"}
                missing = ref_basenames - current_basenames

                if missing:
                    print(f"    [✘] {len(missing)} files missing.")

                if extra:
                    print(f"    [!] {len(extra)} extra files found.")
                    for e in extra:
                        full_filename = current_map[e]
                        file_to_delete = analysis_dir / full_filename
                        if DRY_RUN:
                            print(f"        [DRY RUN] Would delete: {full_filename}")
                        else:
                            try:
                                os.remove(file_to_delete)
                                print(f"        [DELETED] {full_filename}")
                            except OSError as err:
                                print(f"        [ERROR] {full_filename}: {err}")

                if not missing and not extra:
                    print(f"    [✓] Analysis files synced.")
            else:
                print(f"  [!] {category.upper()} Analysis folder missing.")

        print("-" * 45)

    if DRY_RUN:
        print("\n*** DRY RUN COMPLETE. Set DRY_RUN = False to execute. ***")


if __name__ == "__main__":
    cleanup_by_basenames()
