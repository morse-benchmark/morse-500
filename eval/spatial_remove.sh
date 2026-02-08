#!/bin/bash

# Define the list of models (including the commented ones from your list)
models=(
    "Qwen2.5-VL-7B-Instruct"
    "Qwen3-VL-2B-Instruct"
    "Qwen3-VL-2B-Thinking"
    "Qwen3-VL-4B-Instruct"
    "Qwen3-VL-4B-Thinking"
    "Qwen3-VL-8B-Instruct"
    "Qwen3-VL-8B-Thinking"
    "Qwen3-VL-30B-A3B-Instruct-FP8"
    "Qwen3-VL-30B-A3B-Thinking-FP8"
    "Qwen3-VL-235B-A22B-Instruct-FP8"
    "Qwen3-VL-235B-A22B-Thinking-FP8"
)

# Loop through each model
for model in "${models[@]}"; do
    # Check if the model directory itself exists
    if [ ! -d "$model" ]; then
        echo "Skipping: Directory '$model' not found."
        continue
    fi

    # Check if the destination already exists
    if [ -d "$model/spatial_backup" ]; then
        echo "Skipping $model: 'spatial_backup' already exists."
        continue
    fi

    # Check for 'spatial' and rename if found
    if [ -d "$model/spatial" ]; then
        echo "Renaming: $model/spatial -> $model/spatial_backup"
        mv "$model/spatial" "$model/spatial_backup"
    
    # Check for 'spatial_reasoning' and rename if found
    elif [ -d "$model/spatial_reasoning" ]; then
        echo "Renaming: $model/spatial_reasoning -> $model/spatial_backup"
        mv "$model/spatial_reasoning" "$model/spatial_backup"
        
    else
        echo "No change: Neither 'spatial' nor 'spatial_reasoning' found in $model"
    fi
done