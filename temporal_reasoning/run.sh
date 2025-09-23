#!/bin/bash

NUM_DOMINOES=5 python domino_count.py # you can change the NUM_DOMINOES Range 5, 10, 30, 60, 90
NUM_SHAPES=3 python duration_3d.py # you can change the NUM_SHAPES. Range from 3, 5, 7, 10
DIFFICULTY=1 python color_sequence.py # DIFFICULTY can be 1, 2, 3
DIFFICULTY=1 python color_objects.py # DIFFICULTY can be 1, 2, 3
DIFFICULTY=1 python pause_seq.py # DIFFICULTY can be 1 - 5
NUM_BOUNCES=1 python3 bounceball.py # NUM_BOUNCES [1-5] range
NUM_SHAPES=2 python duration2D.py # NUM_SHAPES [2 - 8] range 
NUM_TRANSFORMS=2 python3 color_change.py # NUM_TRANSFORMS [3 - 8] range
NUM_SHAPES=5 python3 numShape.py # NUM_SHAPES [3-8] range

# Configuration
REPEAT_TIMES=1

# Domino count experiments (Range: 5, 10, 30, 60, 90)
echo "Running domino count experiments..."
for dominoes in 5 10 30 60 90; do
    echo "Testing with $dominoes dominoes ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_DOMINOES=$dominoes python domino_count.py
    done
done

# Duration 3D experiments (Range: 3, 5, 7, 10)
echo "Running duration 3D experiments..."
for shapes in 3 5 7 10; do
    echo "Testing with $shapes shapes ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_SHAPES=$shapes python duration_3d.py
    done
done

# Color sequence experiments (Difficulty: 1, 2, 3)
echo "Running color sequence experiments..."
for difficulty in 1 2 3; do
    echo "Testing difficulty $difficulty ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        DIFFICULTY=$difficulty python color_sequence.py
    done
done

# Color objects experiments (Difficulty: 1, 2, 3)
echo "Running color objects experiments..."
for difficulty in 1 2 3; do
    echo "Testing difficulty $difficulty ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        DIFFICULTY=$difficulty python color_objects.py
    done
done

# Pause sequence experiments (Difficulty: 1-5)
echo "Running pause sequence experiments..."
for difficulty in 1 2 3 4 5; do
    echo "Testing difficulty $difficulty ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        DIFFICULTY=$difficulty python pause_seq.py
    done
done