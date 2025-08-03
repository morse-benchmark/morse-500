#!/bin/bash

# Damped SHM Quiz Runner Script
# This script runs the damped_shm_quiz.py with different parameters

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
    echo "  easy   - 1 oscillator"
    echo "  medium - 2 oscillators"
    echo "  hard   - 3 oscillators"
    echo ""
    echo "Examples:"
    echo "  $0                        # Run with default level (medium = 2 oscillators)"
    echo "  $0 -l easy                # Run with easy level (1 oscillator)"
    echo "  $0 --level hard           # Run with hard level (3 oscillators)"
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
        OBJECT_COUNT=1
        ;;
    medium)
        OBJECT_COUNT=2
        ;;
    hard)
        OBJECT_COUNT=3
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
OUTPUT_FILENAME="damped_shm_${LEVEL}_level"

# Display run information
echo "=== Damped SHM Quiz Runner ==="
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
manim -pql damped_shm_quiz.py DampedOscillatorQuizRefactored --output_file "$OUTPUT_FILENAME"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Quiz generation completed successfully!"
else
    echo ""
    echo "❌ Quiz generation failed!"
    exit 1
fi 