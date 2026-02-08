#!/bin/bash

# NUM_DOMINOES=5 python domino_count.py # you can change the NUM_DOMINOES Range 5, 10, 30, 60, 90
# NUM_SHAPES=3 python duration_3d.py # you can change the NUM_SHAPES. Range from 3, 5, 7, 10
# DIFFICULTY=1 python color_sequence.py # DIFFICULTY can be 1, 2, 3
# DIFFICULTY=1 python color_objects.py # DIFFICULTY can be 1, 2, 3
# DIFFICULTY=1 python pause_seq.py # DIFFICULTY can be 1 - 5
# NUM_BOUNCES=1 python3 bounce_ball.py # NUM_BOUNCES [1-5] range
# NUM_SHAPES=2 python duration_2d.py # NUM_SHAPES [2 - 8] range 
# NUM_TRANSFORMS=4 python3 color_change.py # NUM_TRANSFORMS [4 - 8] range
# NUM_SHAPES=3 python3 num_shape.py # NUM_SHAPES [3-8] range

# NUM_SHUFFLES=3 python anagram_all_distance.py # number of shuffles. 3, 5, 7, 9
# NUM_SHUFFLES=3 python anagram_partial_distance.py # number of shuffles. 3, 5, 7, 9
# NUM_SHUFFLES=3 python anagram_position.py # number of shuffles. 3, 5, 7, 9
# NUM_SHUFFLES=3 python anagram_num_shuffles.py # number of shuffles. 3, 5, 7, 9

# # Configuration
REPEAT_TIMES=2

# Domino count experiments (Range: 5, 10, 15, 20, 25, 30, 35, 40)
echo "Running domino count experiments..."
for dominoes in 5 10 20 30 40; do
    echo "Testing with $dominoes dominoes ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_DOMINOES=$dominoes python domino_count.py
    done
done

# Duration 3D experiments (Range: 3, 5, 7, 9, 12)
echo "Running duration 3D experiments..."
for shapes in 3 5 7 9 12; do
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


# Anagram all distance experiments (NUM_SHUFFLES: 3, 5, 7, 9, 12)
echo "Running anagram all distance experiments..."
for shuffles in 3 5 7 9; do
    echo "Testing with $shuffles shuffles ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_SHUFFLES=$shuffles python anagram_all_distance.py
    done
done

# Anagram partial distance experiments (NUM_SHUFFLES: 3, 5, 7, 9, 12)
echo "Running anagram partial distance experiments..."
for shuffles in 3 5 7 9 12; do
    echo "Testing with $shuffles shuffles ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_SHUFFLES=$shuffles python anagram_partial_distance.py
    done
done

# Anagram num shuffles experiments (NUM_SHUFFLES: 3, 5, 7, 9, 12)
echo "Running anagram num shuffles experiments..."
for shuffles in 3 5 7 9 12; do
    echo "Testing with $shuffles shuffles ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_SHUFFLES=$shuffles python anagram_num_shuffles.py
    done
done

# Anagram position experiments (NUM_SHUFFLES: 3, 5, 7, 9, 12)
echo "Running anagram position experiments..."
for shuffles in 3 5 7 9 12; do
    echo "Testing with $shuffles shuffles ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_SHUFFLES=$shuffles python anagram_position.py
    done
done

# Color change experiments (NUM_TRANSFORMS: 4, 5, 6, 7, 8)
echo "Running color change experiments..."
for transforms in 4 5 6 7 8; do
    echo "Testing with $transforms transforms ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_TRANSFORMS=$transforms python color_change.py
    done
done

# Bounce ball experiments (NUM_BOUNCES: 1, 2, 3, 4, 5)
echo "Running bounce ball experiments..."
for bounces in 1 2 3 4 5; do
    echo "Testing with $bounces bounces ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_BOUNCES=$bounces python bounce_ball.py
    done
done

# Duration 2D experiments (NUM_SHAPES: 2, 3, 4, 5, 6, 7, 8)
echo "Running duration 2D experiments..."
for shapes in 2 3 4 5 6; do
    echo "Testing with $shapes shapes ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_SHAPES=$shapes python duration_2d.py
    done
done

# Num shape experiments (NUM_SHAPES: 3, 4, 5, 6, 7, 8)
echo "Running num shape experiments..."
for shapes in 3 4 5 6 7; do
    echo "Testing with $shapes shapes ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_SHAPES=$shapes python num_shape.py
    done
done

echo "All experiments completed!"