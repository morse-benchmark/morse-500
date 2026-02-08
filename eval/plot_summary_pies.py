#!/usr/bin/env python3
"""
plot_summary_pies.py

Usage:
  python plot_summary_pies.py --input summary.json --outdir plots --show
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

CATEGORY_NAME = "spatial_reasoning"

# --- NEW: fixed primary category order + fixed color map ---
PRIMARY_CATEGORY_ORDER = [
    "Hallucinations",
    "Bad Reasoning / Logic",
    "Mathematical/Numerical Errors",
    "Temporal and Sequential Reasoning Errors",
    "Visual and Perceptual Misinterpretation",
]

# Use a stable categorical colormap. tab10 gives 10 distinct colors.
_PRIMARY_CMAP = plt.get_cmap("tab10")
PRIMARY_CATEGORY_COLORS = {
    name: _PRIMARY_CMAP(i) for i, name in enumerate(PRIMARY_CATEGORY_ORDER)
}
FALLBACK_COLOR = (0.7, 0.7, 0.7, 1.0)  # gray for unknown categories


def sanitize_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "untitled"


def autopct_from_values(values: List[float], min_pct_to_label: float = 3.0):
    total = sum(values)

    def _fmt(pct: float) -> str:
        if pct < min_pct_to_label or total <= 0:
            return ""
        val = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({val})"

    return _fmt


def plot_pie(
    labels: List[str],
    values: List[float],
    title: str,
    outpath: str,
    min_pct_to_label: float = 3.0,
    colors: Optional[List] = None,  # --- NEW ---
):
    if not values or sum(values) <= 0:
        print(f"[WARN] Skipping '{title}' (no positive values).")
        return

    plt.figure(figsize=(9, 9))
    plt.pie(
        values,
        labels=labels,
        autopct=autopct_from_values(values, min_pct_to_label=min_pct_to_label),
        startangle=90,
        counterclock=False,
        colors=colors,  # --- NEW ---
    )
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved: {outpath}")


def load_summary(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_subcategories_for_primary(
    primary: str, subcat_dist: Dict[str, int]
) -> List[Tuple[str, int]]:
    prefix = f"{primary} - "
    items = []
    for k, v in subcat_dist.items():
        if isinstance(k, str) and k.startswith(prefix):
            sub_label = k[len(prefix) :].strip()
            items.append((sub_label, int(v)))
    items.sort(key=lambda x: x[1], reverse=True)
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="Qwen3-VL-8B-Instruct",
        help="Model corresponding to summary.json",
    )
    parser.add_argument("--input", default=None, help="Path to summary.json")
    parser.add_argument("--outdir", default="plots", help="Output directory for PNGs")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (in addition to saving).",
    )
    parser.add_argument(
        "--min-pct-label",
        type=float,
        default=3.0,
        help="Hide pie labels below this percentage (default: 3.0).",
    )
    args = parser.parse_args()

    input_filename = (
        args.input
        if args.input
        else f"{args.model_name}/{CATEGORY_NAME}/analysis/summary.json"
    )

    data = load_summary(input_filename)

    primary_dist = data.get("primary_distribution", {})
    subcat_dist = data.get("subcategory_distribution", {})

    if not isinstance(primary_dist, dict) or not isinstance(subcat_dist, dict):
        raise ValueError(
            "summary.json must contain dicts for primary_distribution and subcategory_distribution."
        )

    # 1) Primary distribution pie
    if "None" in primary_dist:
        del primary_dist["None"]

    primary_items = [(k, int(v)) for k, v in primary_dist.items()]
    primary_items.sort(key=lambda x: x[1], reverse=True)

    primary_labels = [k for k, _ in primary_items]
    primary_values = [v for _, v in primary_items]

    # --- NEW: fixed colors for primary pie, keyed by category name ---
    primary_colors = [
        PRIMARY_CATEGORY_COLORS.get(lbl, FALLBACK_COLOR) for lbl in primary_labels
    ]

    plot_pie(
        labels=primary_labels,
        values=primary_values,
        title="Primary Distribution",
        outpath=os.path.join(
            args.outdir, f"{args.model_name}_primary_distribution_pie.png"
        ),
        min_pct_to_label=args.min_pct_label,
        colors=primary_colors,  # --- NEW ---
    )

    # 2) One pie per primary category for its subcategories (unchanged coloring)
    for primary_name, primary_count in primary_items:
        sub_items = extract_subcategories_for_primary(primary_name, subcat_dist)

        if not sub_items:
            print(f"[INFO] No subcategories found for primary '{primary_name}'.")
            continue

        sub_labels = [lbl for lbl, _ in sub_items]
        sub_values = [cnt for _, cnt in sub_items]

        safe = sanitize_filename(primary_name)
        plot_pie(
            labels=sub_labels,
            values=sub_values,
            title=f"Subcategory Distribution: {primary_name}",
            outpath=os.path.join(
                args.outdir, f"{args.model_name}_subcategories_{safe}.png"
            ),
            min_pct_to_label=args.min_pct_label,
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
