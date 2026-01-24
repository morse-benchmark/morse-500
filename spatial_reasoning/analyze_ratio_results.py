#!/usr/bin/env python3
"""
Ratio Results Analyzer
======================

Analyzes evaluation results from ratio sweep to provide:
1. Accuracy vs ratio curves
2. Difficulty progression analysis
3. Step size recommendations
4. Custom accuracy breakdowns

Usage:
    python analyze_ratio_results.py --results results_cubes_count_*.json
"""

import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import numpy as np


def load_results(results_file: Path) -> Dict:
    """Load results from JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_accuracy_curve(results: Dict, output_file: Path):
    """Plot accuracy vs ratio curve"""
    ratios = []
    accuracies = []

    for ratio_str, metrics in sorted(results['results'].items(), key=lambda x: float(x[0])):
        ratios.append(float(ratio_str))
        accuracies.append(metrics['accuracy'] * 100)  # Convert to percentage

    plt.figure(figsize=(12, 6))
    plt.plot(ratios, accuracies, 'bo-', linewidth=2, markersize=8, label='Accuracy')

    # Add threshold line at 90%
    plt.axhline(y=90, color='r', linestyle='--', linewidth=1, label='90% threshold')

    # Highlight threshold ratio
    if results.get('threshold_ratio'):
        threshold_idx = ratios.index(results['threshold_ratio'])
        plt.plot(results['threshold_ratio'], accuracies[threshold_idx],
                'go', markersize=12, label=f'Threshold: {results["threshold_ratio"]}%')

    plt.xlabel('Screen Area Ratio (%)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f"Accuracy vs Size Ratio\n{results['model']} - {results['program']} ({results['p_type']})",
             fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✓ Plot saved to {output_file}")

    plt.close()


def calculate_difficulty_metrics(results: Dict) -> Dict:
    """Calculate difficulty progression metrics"""

    ratios = []
    accuracies = []

    for ratio_str, metrics in sorted(results['results'].items(), key=lambda x: float(x[0])):
        ratios.append(float(ratio_str))
        accuracies.append(metrics['accuracy'])

    ratios = np.array(ratios)
    accuracies = np.array(accuracies)

    # Find difficulty levels
    easy_ratios = ratios[accuracies >= 0.95]
    medium_ratios = ratios[(accuracies >= 0.8) & (accuracies < 0.95)]
    hard_ratios = ratios[accuracies < 0.8]

    # Calculate rate of difficulty increase (accuracy drop per ratio unit)
    if len(ratios) > 1:
        accuracy_derivatives = np.diff(accuracies) / np.diff(ratios)
        avg_difficulty_increase = -np.mean(accuracy_derivatives)  # Negative because accuracy decreases
    else:
        avg_difficulty_increase = 0

    # Find the "knee" - point of steepest descent
    if len(accuracy_derivatives) > 0:
        knee_idx = np.argmin(accuracy_derivatives)
        knee_ratio = ratios[knee_idx]
    else:
        knee_ratio = None

    return {
        "easy_range": (float(easy_ratios.min()), float(easy_ratios.max())) if len(easy_ratios) > 0 else None,
        "medium_range": (float(medium_ratios.min()), float(medium_ratios.max())) if len(medium_ratios) > 0 else None,
        "hard_range": (float(hard_ratios.min()), float(hard_ratios.max())) if len(hard_ratios) > 0 else None,
        "avg_difficulty_increase_per_percent": float(avg_difficulty_increase),
        "knee_ratio": float(knee_ratio) if knee_ratio is not None else None,
        "max_achievable_ratio": results.get('threshold_ratio'),
    }


def recommend_step_sizes(results: Dict, difficulty_metrics: Dict) -> Dict:
    """Recommend step sizes for different difficulty ranges"""

    recommendations = {}

    # Easy range: larger steps
    if difficulty_metrics['easy_range']:
        easy_min, easy_max = difficulty_metrics['easy_range']
        easy_step = max(1.0, (easy_max - easy_min) / 3)  # 3-4 samples in easy range
        recommendations['easy'] = {
            'range': difficulty_metrics['easy_range'],
            'recommended_step': round(easy_step, 1),
            'ratios': list(np.arange(easy_min, easy_max + easy_step, easy_step))
        }

    # Medium range: moderate steps
    if difficulty_metrics['medium_range']:
        med_min, med_max = difficulty_metrics['medium_range']
        med_step = max(0.5, (med_max - med_min) / 5)  # 5-6 samples in medium range
        recommendations['medium'] = {
            'range': difficulty_metrics['medium_range'],
            'recommended_step': round(med_step, 1),
            'ratios': list(np.arange(med_min, med_max + med_step, med_step))
        }

    # Hard range: fine-grained steps
    if difficulty_metrics['hard_range']:
        hard_min, hard_max = difficulty_metrics['hard_range']
        hard_step = max(0.5, (hard_max - hard_min) / 8)  # 8-10 samples in hard range
        recommendations['hard'] = {
            'range': difficulty_metrics['hard_range'],
            'recommended_step': round(hard_step, 1),
            'ratios': list(np.arange(hard_min, hard_max + hard_step, hard_step))
        }

    return recommendations


def generate_custom_accuracy_table(results: Dict) -> str:
    """Generate custom accuracy table formatted for easy reading"""

    lines = []
    lines.append("\n" + "="*80)
    lines.append("CUSTOM ACCURACY TABLE")
    lines.append("="*80)
    lines.append(f"Model: {results['model']}")
    lines.append(f"Program: {results['program']} ({results['p_type']})")
    lines.append(f"Samples per ratio: {results['samples_per_ratio']}")
    lines.append("="*80)
    lines.append("")

    lines.append(f"{'Ratio (%)':<12} {'Accuracy':<12} {'Correct':<10} {'Total':<10} {'Category':<12}")
    lines.append("-"*56)

    for ratio_str, metrics in sorted(results['results'].items(), key=lambda x: float(x[0])):
        ratio = float(ratio_str)
        acc = metrics['accuracy']
        correct = metrics['correct']
        total = metrics['total']

        # Categorize
        if acc >= 0.95:
            category = "Easy"
        elif acc >= 0.8:
            category = "Medium"
        elif acc >= 0.5:
            category = "Hard"
        else:
            category = "Very Hard"

        lines.append(f"{ratio:<12.1f} {acc:<12.1%} {correct:<10} {total:<10} {category:<12}")

    lines.append("")
    lines.append("="*80)

    return "\n".join(lines)


def generate_benchmark_config(results: Dict, difficulty_metrics: Dict,
                              step_recommendations: Dict) -> Dict:
    """Generate recommended benchmark configuration"""

    config = {
        "program": results['program'],
        "p_type": results['p_type'],
        "model_evaluated": results['model'],
        "max_achievable_ratio": difficulty_metrics['max_achievable_ratio'],
        "difficulty_ranges": {
            "easy": difficulty_metrics['easy_range'],
            "medium": difficulty_metrics['medium_range'],
            "hard": difficulty_metrics['hard_range'],
        },
        "recommended_test_ratios": {}
    }

    # Collect all recommended test ratios
    all_ratios = []
    for difficulty, rec in step_recommendations.items():
        config['recommended_test_ratios'][difficulty] = rec['ratios']
        all_ratios.extend(rec['ratios'])

    # Sort and deduplicate
    config['recommended_test_ratios']['all'] = sorted(list(set(all_ratios)))

    return config


def main():
    parser = argparse.ArgumentParser(description="Analyze ratio sweep results")
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results JSON file')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for plots and analysis')

    args = parser.parse_args()

    results_file = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Ratio Results Analysis")
    print(f"{'='*80}\n")

    # Load results
    print(f"Loading results from {results_file}...")
    results = load_results(results_file)

    # Generate accuracy plot
    plot_file = output_dir / f"accuracy_curve_{results['program']}_{results['p_type']}.png"
    print(f"\nGenerating accuracy curve...")
    plot_accuracy_curve(results, plot_file)

    # Calculate difficulty metrics
    print(f"\nCalculating difficulty metrics...")
    difficulty_metrics = calculate_difficulty_metrics(results)

    print(f"\nDifficulty Analysis:")
    print(f"  Easy range (≥95% acc): {difficulty_metrics['easy_range']}")
    print(f"  Medium range (80-95% acc): {difficulty_metrics['medium_range']}")
    print(f"  Hard range (<80% acc): {difficulty_metrics['hard_range']}")
    print(f"  Avg difficulty increase: {difficulty_metrics['avg_difficulty_increase_per_percent']:.3f} acc/ratio%")
    print(f"  Knee point (steepest drop): {difficulty_metrics['knee_ratio']}")
    print(f"  Max achievable (≥90% acc): {difficulty_metrics['max_achievable_ratio']}")

    # Generate step size recommendations
    print(f"\nGenerating step size recommendations...")
    step_recommendations = recommend_step_sizes(results, difficulty_metrics)

    print(f"\nRecommended Step Sizes:")
    for difficulty, rec in step_recommendations.items():
        print(f"  {difficulty.capitalize()}:")
        print(f"    Range: {rec['range']}")
        print(f"    Step: {rec['recommended_step']}%")
        print(f"    Test ratios: {rec['ratios'][:5]}..." if len(rec['ratios']) > 5 else f"    Test ratios: {rec['ratios']}")

    # Generate custom accuracy table
    print(f"\nGenerating custom accuracy table...")
    accuracy_table = generate_custom_accuracy_table(results)
    print(accuracy_table)

    # Save accuracy table
    table_file = output_dir / f"accuracy_table_{results['program']}_{results['p_type']}.txt"
    with open(table_file, 'w') as f:
        f.write(accuracy_table)
    print(f"\n✓ Accuracy table saved to {table_file}")

    # Generate benchmark config
    print(f"\nGenerating benchmark configuration...")
    benchmark_config = generate_benchmark_config(results, difficulty_metrics, step_recommendations)

    config_file = output_dir / f"benchmark_config_{results['program']}_{results['p_type']}.json"
    with open(config_file, 'w') as f:
        json.dump(benchmark_config, f, indent=2)
    print(f"✓ Benchmark config saved to {config_file}")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {results['model']}")
    print(f"Program: {results['program']} ({results['p_type']})")
    print(f"Max achievable ratio: {difficulty_metrics['max_achievable_ratio']}%")
    print(f"Recommended test ratios: {len(benchmark_config['recommended_test_ratios']['all'])}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
