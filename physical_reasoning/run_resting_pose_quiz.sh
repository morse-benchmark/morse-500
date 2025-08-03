#!/bin/bash

# Resting Pose Quiz Runner Script
# This script runs the resting_pose.py with different parameters

# Default values
DEFAULT_LEVEL="medium"

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -l, --level LEVEL         Set difficulty level: easy/medium/hard (default: $DEFAULT_LEVEL)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Difficulty Levels:"
    echo "  easy   - 3 oscillators"
    echo "  medium - 4 oscillators"
    echo "  hard   - 5 oscillators"
    echo ""
    echo "Examples:"
    echo "  $0                        # Run with default level (medium = 4 oscillators)"
    echo "  $0 -l easy                # Run with easy level (3 oscillators)"
    echo "  $0 --level hard           # Run with hard level (5 oscillators)"
    echo ""
}

# Parse command line arguments
LEVEL=$DEFAULT_LEVEL

while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--level)
            LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Convert level to object count
case $LEVEL in
    easy)
        OBJECT_COUNT=3
        ;;
    medium)
        OBJECT_COUNT=4
        ;;
    hard)
        OBJECT_COUNT=5
        ;;
    *)
        echo "Error: Level must be 'easy', 'medium', or 'hard'"
        exit 1
        ;;
esac

# Generate random seed and difficulty
SEED=$((RANDOM * RANDOM))  # Generate a larger random seed
DIFFICULTY=$((RANDOM % 10))  # Random difficulty 0-9

# Create filename based on level
OUTPUT_FILENAME="resting_pose_${LEVEL}_level"

# Display run information
echo "=== Resting Pose Quiz Runner ==="
echo "Seed: $SEED (random)"
echo "Difficulty: $DIFFICULTY (random 0-9)"
echo "Level: $LEVEL"
echo "Object Count: $OBJECT_COUNT"
echo "Output Filename: $OUTPUT_FILENAME"
echo "================================"

# Set environment variables and run manim
export MANIM_SEED=$SEED
export MANIM_DIFFICULTY=$DIFFICULTY
export MANIM_OBJECT_COUNT=$OBJECT_COUNT

# Run the manim command from current directory with custom output filename
manim -pql resting_pose.py DampedOscillatorsRestingPositionQuiz --output_file "$OUTPUT_FILENAME"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Quiz generation completed successfully!"
else
    echo ""
    echo "❌ Quiz generation failed!"
    exit 1
fi 