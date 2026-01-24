# Ratio-Based Size Parameter - Quick Start Guide

## Overview

The ratio-based system replaces abstract difficulty parameters (0.0-1.0) with **screen area ratio** - the percentage of screen pixels occupied by the visual elements.

This provides:
- ✅ Interpretable difficulty metric
- ✅ Consistent across different problem types
- ✅ Easy to reason about and tune
- ✅ Direct relationship to visual complexity

## Key Concepts

### Screen Area Ratio (%)

The `SIZE_RATIO` parameter represents what percentage of the 1920×1080 screen (2,073,600 pixels) is occupied by the visual elements:

- **1-5%**: Very easy, tiny elements
- **10%**: Easy-medium, clearly visible
- **15-20%**: Medium difficulty
- **25-30%**: Challenging
- **35%+**: Very hard

### Programs

Three programs support ratio-based generation:

1. **cubes_ratio.py** - 3D cube counting
2. **ropes_ratio.py** - Line intersection puzzles
3. **domino_count_ratio.py** - Sequential counting

## Quick Usage

### 1. Generate Videos at Specific Ratio

```bash
# Generate a cube counting video at 10% screen coverage
export SIZE_RATIO=10.0
export P_TYPE=count
python cubes_ratio.py

# Result: questions/cubes_ratio_count_r10.0_seed1234.mp4
```

### 2. Generate Multiple Ratios

```bash
# Test easy to hard progression
for ratio in 5 10 15 20 25; do
    export SIZE_RATIO=$ratio
    python cubes_ratio.py
done
```

### 3. Run Full Evaluation Sweep

```bash
# Evaluate model across ratio spectrum
python eval_ratio_sweep.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --port 8000 \
    --program cubes \
    --p_type count \
    --ratios "1,2,5,10,15,20,25,30" \
    --samples_per_ratio 10

# This will:
# 1. Generate 10 videos at each ratio level
# 2. Evaluate all with the model
# 3. Calculate accuracy at each level
# 4. Find highest ratio before failure
# 5. Save results to JSON
```

### 4. Analyze Results

```bash
# Generate accuracy curves and recommendations
python analyze_ratio_results.py \
    --results results_cubes_count_Qwen_Qwen3-VL-8B-Instruct.json

# Outputs:
# - accuracy_curve_cubes_count.png (plot)
# - accuracy_table_cubes_count.txt (detailed table)
# - benchmark_config_cubes_count.json (recommended test ratios)
```

## Understanding Results

### Accuracy Table

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

### Difficulty Categories

- **Easy (≥95% accuracy)**: Model handles reliably
- **Medium (80-95%)**: Model mostly correct with some errors
- **Hard (50-80%)**: Model struggles significantly
- **Very Hard (<50%)**: Model mostly fails

### Key Metrics

- **Max Achievable Ratio**: Highest ratio with ≥90% accuracy (threshold for "passing")
- **Knee Point**: Ratio where accuracy drops most steeply (difficulty transition)
- **Step Size**: Recommended spacing between test ratios for that difficulty range

## Example Workflow

### Full Model Benchmarking

```bash
#!/bin/bash

MODEL="Qwen/Qwen3-VL-8B-Instruct"
PORT=8000

# 1. Run sweep for all programs
for program in cubes ropes domino_count; do
    python eval_ratio_sweep.py \
        --model $MODEL \
        --port $PORT \
        --program $program \
        --ratios "1,2,5,10,15,20,25,30" \
        --samples_per_ratio 10
done

# 2. Analyze each
for program in cubes ropes domino_count; do
    python analyze_ratio_results.py \
        --results results_${program}_count_*.json
done

# 3. Compare max achievable ratios
grep "Max achievable" results_*.json
```

## Tips

### Choosing Ratios

- Start with wide spacing: `1, 5, 10, 20, 30`
- Refine around the failure point with finer steps
- Use analysis tool recommendations for optimal spacing

### Interpreting Failures

- **No response**: Model timed out or crashed
- **No answer extracted**: Model didn't format answer correctly (missing `\boxed{}`)
- **Incorrect**: Model provided wrong answer

### Optimizing Generation

- Small ratios (1-10%) generate quickly (~10s per video)
- Large ratios (25%+) take longer (~30s per video)
- Use `--samples_per_ratio` to balance accuracy vs time

## Files Generated

### Per Video

- `questions/program_ratio_ptype_rX.X_seedYYYY.mp4` - Question video
- `solutions/program_ratio_ptype_rX.X_seedYYYY.txt` - Ground truth answer
- `question_text/program_ratio_ptype_rX.X_seedYYYY.txt` - Question text
- `reasoning_traces/program_ratio_ptype_rX.X_seedYYYY.txt` - Step-by-step solution

### Per Evaluation

- `results_program_ptype_model.json` - Raw evaluation results
- `accuracy_curve_program_ptype.png` - Accuracy vs ratio plot
- `accuracy_table_program_ptype.txt` - Detailed accuracy breakdown
- `benchmark_config_program_ptype.json` - Recommended test configuration

## Common Issues

### Video Generation Fails

```bash
# Check if Manim is installed
python -c "import manim; print(manim.__version__)"

# Check environment variables
echo $SIZE_RATIO
echo $P_TYPE
```

### Evaluation Runs Out of Memory

```bash
# Reduce concurrent requests
python eval_ratio_sweep.py --max_concurrent 2

# Reduce video resolution
python eval_ratio_sweep.py --max_size 256
```

### Model Not Found

```bash
# Check vLLM server is running
curl http://localhost:8000/v1/models

# Check port matches
python eval_ratio_sweep.py --port 8000  # default
```

## Next Steps

1. **Generate baseline**: Run sweeps for all programs
2. **Find thresholds**: Identify max achievable ratios
3. **Refine ranges**: Generate additional samples near boundaries
4. **Compare models**: Run same sweeps with different models
5. **Build benchmark**: Use recommended configs to create standardized test suite

## See Also

- `RATIO_PARAMETER_REFERENCE.md` - Complete technical documentation
- `eval_ratio_sweep.py --help` - Full CLI options
- `analyze_ratio_results.py --help` - Analysis options
