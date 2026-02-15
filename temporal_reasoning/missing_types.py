import os
import re
from collections import Counter


def find_missing_seeds(directory=".", target_count=10):
    # Regex to capture the type (e.g., numshape_n3) and the seed
    # It looks for "anything_seedNUMBER.mp4"
    pattern = re.compile(r"(.+)_seed\d+\.mp4$")

    video_types = []

    # List files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            video_type = match.group(1)
            video_types.append(video_type)

    # Count occurrences of each type
    counts = Counter(video_types)

    # Report results
    print(f"{'Video Type':<25} | {'Count':<6} | {'Status'}")
    print("-" * 45)

    missing_any = False
    for v_type, count in sorted(counts.items()):
        if count < target_count:
            status = f"MISSING {target_count - count}"
            print(f"{v_type:<25} | {count:<6} | {status}")
            missing_any = True

    if not missing_any:
        print("All video types have at least 10 seeds!")


if __name__ == "__main__":
    find_missing_seeds()
