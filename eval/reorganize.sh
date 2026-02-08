#!/bin/bash

# Array of folder names from the image
folders=(
    "Qwen3-VL-2B-Instruct"
    "Qwen3-VL-2B-Instruct_analyze"
    "Qwen3-VL-2B-Thinking"
    "Qwen3-VL-2B-Thinking_analyze"
    "Qwen3-VL-4B-Instruct"
    "Qwen3-VL-4B-Instruct_analyze"
    "Qwen3-VL-4B-Thinking"
    "Qwen3-VL-4B-Thinking_analyze"
    "Qwen3-VL-8B-Instruct"
    "Qwen3-VL-8B-Instruct_analyze"
    "Qwen3-VL-8B-Thinking"
    "Qwen3-VL-8B-Thinking_analyze"
    "Qwen3-VL-8B-Instruct-FP8"
    "Qwen3-VL-30B-A3B-Instruct-FP8"
    "Qwen3-VL-30B-A3B-Instruct-FP8_analyze"
    "Qwen3-VL-30B-A3B-Thinking-FP8"
    "Qwen3-VL-30B-A3B-Thinking-FP8_analyze"
    "Qwen3-VL-32B-Instruct-FP8"
    "Qwen3-VL-32B-Thinking-FP8"
    "Qwen3-VL-235B-A22B-Instruct-FP8"
    "Qwen3-VL-235B-A22B-Instruct-FP8_analyze"
    "Qwen3-VL-235B-A22B-Thinking-FP8"
    "Qwen3-VL-235B-A22B-Thinking-FP8_analyze"
)

# Loop through each folder
for folder in "${folders[@]}"; do
    if [ -d "$folder" ]; then
        echo "Processing: $folder"
        
        # Create spatial subfolder if it doesn't exist
        mkdir -p "$folder/spatial"
        
        # Move all files (not directories) to spatial subfolder
        find "$folder" -maxdepth 1 -type f -exec mv {} "$folder/spatial/" \;
        
        echo "✓ Completed: $folder"
    else
        echo "✗ Folder not found: $folder"
    fi
done

echo "All done!"