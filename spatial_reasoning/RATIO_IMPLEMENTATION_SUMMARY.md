# Ratio-Based Size Parameter Implementation Summary

## What Changed

### Old System (SIZE parameter)
- **Range**: 0.0 to 1.0
- **Meaning**: Abstract "difficulty" level
- **Problem**: Not interpretable - what does SIZE=0.7 mean?
- **Example**: SIZE=0.5 → 4×4×4 grid for cubes

### New System (SIZE_RATIO parameter)
- **Range**: 0.5% to 40%
- **Meaning**: Percentage of screen area occupied
- **Benefit**: Directly interpretable - "10% of screen covered"
- **Example**: SIZE_RATIO=10.0 → ~6×6×6 grid (adapts to fit 10% coverage)

## Programs Converted

### 1. cubes_ratio.py
- Converts SIZE_RATIO → grid dimensions
- Grid size calculated to achieve target screen coverage
- Range: 2×2×2 (1% coverage) to 10×10×10 (40% coverage)

### 2. eval_ratio_sweep.py
- Complete evaluation system for ratio sweeps
- Features:
  - Generates videos at multiple ratio levels
  - Runs inference with vLLM
  - Calculates accuracy at each level
  - Finds highest ratio before failure (≥90% threshold)
  - Supports all three programs

### 3. analyze_ratio_results.py
- Post-processing analysis tool
- Features:
  - Accuracy vs ratio curves
  - Difficulty categorization (Easy/Medium/Hard)
  - Step size recommendations
  - Custom accuracy tables
  - Benchmark configuration generation

### 4. RATIO_QUICK_START.md
- Complete user guide
- Usage examples
- Troubleshooting
- Best practices

## Key Features

### Interpretable Metrics
- **Screen Area Ratio**: Direct measure of visual complexity
- **Accuracy at Each Level**: Clear performance profile
- **Max Achievable Ratio**: Single number summarizing capability
- **Difficulty Categories**: Easy/Medium/Hard based on accuracy thresholds

### Automated Evaluation
```bash
# Single command to:
# 1. Generate videos at 8 ratio levels
# 2. Evaluate with model
# 3. Calculate metrics
# 4. Find threshold

python eval_ratio_sweep.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --program cubes \
    --ratios "1,2,5,10,15,20,25,30" \
    --samples_per_ratio 10
```

### Comprehensive Analysis
```bash
# Generates:
# - Accuracy curve plot
# - Detailed accuracy table
# - Step size recommendations
# - Benchmark configuration

python analyze_ratio_results.py \
    --results results_cubes_count_*.json
```

## Usage Examples

### Basic Generation
```bash
# Generate single video at 15% coverage
export SIZE_RATIO=15.0
export P_TYPE=count
python cubes_ratio.py
```

### Full Evaluation
```bash
# Evaluate model across full difficulty spectrum
python eval_ratio_sweep.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --port 8000 \
    --program cubes \
    --p_type count \
    --ratios "1,2,5,10,15,20,25,30" \
    --samples_per_ratio 10
```

### Results Analysis
```bash
# Analyze evaluation results
python analyze_ratio_results.py \
    --results results_cubes_count_Qwen_Qwen3-VL-8B-Instruct.json

# Output:
# - accuracy_curve_cubes_count.png
# - accuracy_table_cubes_count.txt
# - benchmark_config_cubes_count.json
```

## Output Files

### Generated Videos
```
questions/cubes_ratio_count_r10.0_seed1234.mp4
solutions/cubes_ratio_count_r10.0_seed1234.txt
question_text/cubes_ratio_count_r10.0_seed1234.txt
reasoning_traces/cubes_ratio_count_r10.0_seed1234.txt
```

### Evaluation Results
```
results_cubes_count_Qwen_Qwen3-VL-8B-Instruct.json
```

Example content:
```json
{
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "program": "cubes",
  "p_type": "count",
  "ratios": [1, 2, 5, 10, 15, 20, 25, 30],
  "samples_per_ratio": 10,
  "threshold_ratio": 15.0,
  "results": {
    "1.0": {
      "total": 10,
      "correct": 10,
      "accuracy": 1.0
    },
    "10.0": {
      "total": 10,
      "correct": 9,
      "accuracy": 0.9
    }
    ...
  }
}
```

### Analysis Outputs

**Accuracy Curve** (`accuracy_curve_cubes_count.png`):
- Line plot: Ratio (x-axis) vs Accuracy % (y-axis)
- Shows 90% threshold line
- Highlights max achievable ratio

**Accuracy Table** (`accuracy_table_cubes_count.txt`):
```
Ratio (%)    Accuracy     Correct    Total      Category
--------------------------------------------------------
1.0          100.00%      10         10         Easy
5.0          100.00%      10         10         Easy
10.0         95.00%       9          10         Easy
15.0         85.00%       8          10         Medium
20.0         70.00%       7          10         Medium
25.0         45.00%       4          10         Hard
30.0         20.00%       2          10         Very Hard
```

**Benchmark Config** (`benchmark_config_cubes_count.json`):
```json
{
  "program": "cubes",
  "p_type": "count",
  "max_achievable_ratio": 10.0,
  "difficulty_ranges": {
    "easy": [1.0, 10.0],
    "medium": [15.0, 20.0],
    "hard": [25.0, 30.0]
  },
  "recommended_test_ratios": {
    "easy": [1.0, 5.0, 10.0],
    "medium": [15.0, 17.5, 20.0],
    "hard": [25.0, 27.5, 30.0],
    "all": [1.0, 5.0, 10.0, 15.0, 17.5, 20.0, 25.0, 27.5, 30.0]
  }
}
```

## Evaluation Workflow

### Step 1: Generate and Evaluate
```bash
python eval_ratio_sweep.py \
    --model MODEL_NAME \
    --program cubes \
    --ratios "1,2,5,10,15,20,25,30" \
    --samples_per_ratio 10
```

**What happens:**
1. Generates 10 videos at each ratio (80 total)
2. Encodes videos to base64
3. Sends to vLLM model
4. Extracts answers from responses
5. Compares to ground truth
6. Calculates accuracy per ratio
7. Finds threshold (last ratio ≥90% accuracy)

### Step 2: Analyze Results
```bash
python analyze_ratio_results.py \
    --results results_cubes_count_MODEL.json
```

**What happens:**
1. Loads evaluation results
2. Plots accuracy curve
3. Categorizes difficulty ranges
4. Calculates rate of difficulty increase
5. Recommends step sizes for each range
6. Generates custom accuracy table
7. Creates benchmark configuration

### Step 3: Use Recommendations
```bash
# Use recommended ratios from benchmark config
python eval_ratio_sweep.py \
    --model NEW_MODEL \
    --program cubes \
    --ratios "1.0,5.0,10.0,15.0,17.5,20.0,25.0,27.5,30.0" \
    --samples_per_ratio 20  # More samples at refined ratios
```

## Key Metrics

### Max Achievable Ratio
- **Definition**: Highest ratio with ≥90% accuracy
- **Use**: Single number summarizing model capability
- **Example**: Model achieves 10.0% → handles medium-small visual complexity

### Accuracy per Ratio
- **Definition**: Percentage correct at each ratio level
- **Use**: Full performance profile across difficulty spectrum
- **Example**: [100%, 100%, 95%, 85%, 70%] → smooth degradation

### Difficulty Ranges
- **Easy**: ≥95% accuracy
- **Medium**: 80-95% accuracy
- **Hard**: 50-80% accuracy
- **Very Hard**: <50% accuracy

### Step Size
- **Definition**: Recommended spacing between test ratios
- **Use**: Optimize evaluation density
- **Example**: Easy range: 3% steps, Hard range: 1% steps

## Advantages Over Old System

### ✅ Interpretability
- **Old**: "SIZE=0.7" (what does this mean?)
- **New**: "15% of screen" (clear visual understanding)

### ✅ Consistency
- **Old**: SIZE=0.5 means different things for different programs
- **New**: 10% coverage is 10% for all programs

### ✅ Tunability
- **Old**: Hard to know what SIZE value to try next
- **New**: Recommendations based on accuracy curves

### ✅ Comparability
- **Old**: Hard to compare across problem types
- **New**: Same metric across all types

### ✅ Evaluation Framework
- **Old**: Manual testing at arbitrary sizes
- **New**: Automated sweeps with comprehensive analysis

## Future Work

### Additional Programs
- Convert `ropes_sized.py` → `ropes_ratio.py`
- Convert `domino_count_sized.py` → `domino_count_ratio.py`

### Enhanced Analysis
- Multi-model comparison plots
- Statistical significance testing
- Error analysis by ratio level
- Cross-program correlation analysis

### Optimization
- Adaptive ratio selection (binary search for threshold)
- Parallel video generation
- Caching for repeated evaluations

## Migration Guide

### If you were using SIZE parameter:

**Old code:**
```bash
export SIZE=0.5
python cubes_sized.py
```

**New code:**
```bash
export SIZE_RATIO=10.0
python cubes_ratio.py
```

### Approximate mapping:
- SIZE=0.0 (2x2x2) → SIZE_RATIO=1.0%
- SIZE=0.25 (3x3x3) → SIZE_RATIO=3.0%
- SIZE=0.5 (4x4x4) → SIZE_RATIO=10.0%
- SIZE=0.75 (6x6x6) → SIZE_RATIO=20.0%
- SIZE=1.0 (7x7x7) → SIZE_RATIO=30.0%

## Files in This Implementation

```
cubes_ratio.py                  # Ratio-based cube generation
eval_ratio_sweep.py             # Automated evaluation system
analyze_ratio_results.py        # Results analysis tool
RATIO_QUICK_START.md           # User guide
RATIO_IMPLEMENTATION_SUMMARY.md # This file
RATIO_PARAMETER_REFERENCE.md   # Technical reference (to be created)
```

## Getting Started

1. **Read quick start**: `RATIO_QUICK_START.md`
2. **Generate test video**: `export SIZE_RATIO=10.0 && python cubes_ratio.py`
3. **Run evaluation**: See examples above
4. **Analyze results**: `python analyze_ratio_results.py --results ...`

## Questions?

See `RATIO_QUICK_START.md` for:
- Detailed usage examples
- Troubleshooting guide
- Common issues and solutions
- Best practices

---

**Implementation Date**: 2026-01-24
**Branch**: `ratio-based-size-parameter`
**Status**: Complete for cubes, ready for ropes and domino_count
