#!/bin/bash
REPEAT_TIMES=1 # 15
# 150 ID and 100 OOD
# 500 training examples; 250 test - 150 ID and 100 OOD

# test: 15 * 13 

# Domino count experiments (Range: 5, 10, 15, 20, 25, 30, 35, 40)
# total questions: 4 * repeat_times
echo "Running domino count experiments..."
for dominoes in 5 10 20 30; do
    echo "Testing with $dominoes dominoes ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_DOMINOES=$dominoes python domino_count.py
    done
done

# Duration 3D experiments (Range: 3, 5, 7, 9, 12)
# total questions: 4 * repeat_times
echo "Running duration 3D experiments..."
for shapes in 3 5 7 9; do
    echo "Testing with $shapes shapes ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_SHAPES=$shapes python duration_3d.py
    done
done

# Color sequence experiments (Difficulty: 1, 2, 3)
# total questions: 3 * repeat_times
# TODO: convert to color count 
echo "Running color sequence experiments..."
for color_count in 5 8 12 16 20; do
    echo "Generating color sequence with $color_count colors"
    for i in $(seq 1 $REPEAT_TIMES); do
        COLOR_COUNT=$color_count python color_sequence.py
    done
done

# Color objects experiments (Difficulty: 1, 2, 3)
# total questions: 3 * repeat_times
echo "Running color objects experiments..."
for num_shapes in 4 5 6 7 8; do
    echo "Generating puzzle with $num_shapes shapes"
    NUM_SHAPES=$num_shapes python color_objects.py
done


# Pause sequence experiments (Difficulty: 1-5)
# total questions: 4 * repeat_times
echo "Running pause sequence experiments..."
for num_sequences in 3 4 5 6; do
    echo "Testing num_sequences = $num_sequences ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_SEQUENCES=$num_sequences python pause_seq.py
    done
done


# Anagram all distance experiments (NUM_SHUFFLES: 3, 5, 7, 9, 12)
# total questions: 4 * repeat_times
echo "Running anagram all distance experiments..."
for shuffles in 3 5 7 9; do
    echo "Testing with $shuffles shuffles ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_SHUFFLES=$shuffles python anagram_all_distance.py
    done
done

# Anagram partial distance experiments (NUM_SHUFFLES: 3, 5, 7, 9, 12)
# total questions: 4 * repeat_times
echo "Running anagram partial distance experiments..."
for shuffles in 3 5 7 9; do
    echo "Testing with $shuffles shuffles ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_SHUFFLES=$shuffles python anagram_partial_distance.py
    done
done

# Anagram num shuffles experiments (NUM_SHUFFLES: 3, 5, 7, 9, 12)
# total questions: 4 * repeat_times
echo "Running anagram num shuffles experiments..."
for shuffles in 3 5 7 9; do
    echo "Testing with $shuffles shuffles ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_SHUFFLES=$shuffles python anagram_num_shuffles.py
    done
done

# Anagram position experiments (NUM_SHUFFLES: 3, 5, 7, 9, 12)
# total questions: 4 * repeat_times
echo "Running anagram position experiments..."
for shuffles in 3 5 7 9; do
    echo "Testing with $shuffles shuffles ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_SHUFFLES=$shuffles python anagram_position.py
    done
done

# Color change experiments (NUM_TRANSFORMS: 4, 5, 6, 7, 8)
# total questions: 4 * repeat_times
echo "Running color change experiments..."
for transforms in 4 5 6 7; do
    echo "Testing with $transforms transforms ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_TRANSFORMS=$transforms python color_change.py
    done
done

# Bounce ball experiments (NUM_BOUNCES: 1, 2, 3, 4, 5)
# total questions: 4 * repeat_times
echo "Running bounce ball experiments..."
for bounces in 1 2 3 4; do
    echo "Testing with $bounces bounces ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_BOUNCES=$bounces python bounce_ball.py
    done
done

# Duration 2D experiments (NUM_SHAPES: 2, 3, 4, 5, 6, 7, 8)
# total questions: 4 * repeat_times
echo "Running duration 2D experiments..."
for shapes in 2 3 4 5; do
    echo "Testing with $shapes shapes ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_SHAPES=$shapes python duration_2d.py
    done
done

# Num shape experiments (NUM_SHAPES: 3, 4, 5, 6, 7, 8)
# total questions: 4 * repeat_times
echo "Running num shape experiments..."
for shapes in 3 4 5 6; do
    echo "Testing with $shapes shapes ($REPEAT_TIMES times)"
    for i in $(seq 1 $REPEAT_TIMES); do
        NUM_SHAPES=$shapes python num_shape.py
    done
done

echo "All experiments completed!"