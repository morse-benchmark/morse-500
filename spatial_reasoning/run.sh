#!/bin/bash


# MAX_SIZE can be 3, 4, 5+
# P_TYPE=count MAX_SIZE=3 python cubes.py 
# P_TYPE=missing MAX_SIZE=3 python cubes.py 
# P_TYPE=surface_area MAX_SIZE=3 python cubes.py
# P_TYPE=exposed MAX_SIZE=3 python cubes.py
# P_TYPE=colors MAX_SIZE=3 python cubes.py
# P_TYPE=max_color MAX_SIZE=3 python cubes.py
# P_TYPE=project MAX_SIZE=3 python cubes.py
# P_TYPE=missing_shape MAX_SIZE=3 python cubes.py
# P_TYPE=matching MAX_SIZE=3 python cubes.py


# PATH_LENGTH=5 python cube_path.py


# P_TYPE=hidden NUM_DICE=4 N_ROLL=5 python dice.py
# P_TYPE=max_hidden NUM_DICE=4 N_ROLL=5 python dice.py
# P_TYPE=match NUM_DICE=4 N_ROLL=5 python dice.py
# P_TYPE=roll NUM_DICE=4 N_ROLL=5 python dice.py
# P_TYPE=n_roll NUM_DICE=4 N_ROLL=5 python dice.py
# P_TYPE=fold NUM_DICE=4 N_ROLL=5 python dice.py


# # NUM_SHAPES can be 3, 4, 5, 6 +
# P_TYPE=max_dist NUM_SHAPES=3 python path.py
# P_TYPE=min_dist NUM_SHAPES=3 python path.py
# P_TYPE=order NUM_SHAPES=3 python path.py
# P_TYPE=max_time NUM_SHAPES=3 python path.py
# P_TYPE=min_time NUM_SHAPES=3 python path.py


# P_TYPE=count NUM_ROPES=4 BENDS_PER_ROPE=3 python ropes.py
# P_TYPE=cut NUM_ROPES=4 BENDS_PER_ROPE=3 python ropes.py
# P_TYPE=closed NUM_ROPES=4 BENDS_PER_ROPE=3 python ropes.py
# P_TYPE=order NUM_ROPES=4 BENDS_PER_ROPE=3 python ropes.py


# DIFFICULTY=1 python cubenet.py # difficulty level. 1, 2, 3, 4, 5


# # can go over 5 for more complex cubes
# for max_size in 3 4 5; do
#     echo "Testing with MAX_SIZE=$max_size"
    
#     P_TYPE=count MAX_SIZE=3 python cubes.py 
#     P_TYPE=missing MAX_SIZE=3 python cubes.py 
#     P_TYPE=surface_area MAX_SIZE=3 python cubes.py
#     P_TYPE=exposed MAX_SIZE=3 python cubes.py
#     P_TYPE=colors MAX_SIZE=3 python cubes.py
#     P_TYPE=max_color MAX_SIZE=3 python cubes.py
#     P_TYPE=project MAX_SIZE=3 python cubes.py
#     P_TYPE=missing_shape MAX_SIZE=3 python cubes.py
#     P_TYPE=matching MAX_SIZE=3 python cubes.py
    
#     echo "Completed MAX_SIZE=$max_size experiments"
#     echo ""
# done

# NUM_SHUFFLES=3 python anagram_all_distance.py # number of shuffles. 3, 5, 7, 9
# NUM_SHUFFLES=3 python anagram_partial_distance.py # number of shuffles. 3, 5, 7, 9
# NUM_SHUFFLES=3 python anagram_position.py # number of shuffles. 3, 5, 7, 9
# NUM_SHUFFLES=3 python anagram_num_shuffles.py # number of shuffles. 3, 5, 7, 9


#!/bin/bash

# Function to retry command until a file with the specified prefix is generated
retry_until_file_exists() {
    local cmd="$1"
    local file_prefix="$2"
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt: $cmd"
        eval "$cmd" 2>&1
        
        # Check if any file with the prefix exists in questions folder
        if ls questions/${file_prefix}* 1> /dev/null 2>&1; then
            echo "Success! Found file: questions/${file_prefix}*"
            break
        else
            echo "Failed: No file found with prefix 'questions/${file_prefix}*', retrying..."
            ((attempt++))
        fi
    done
    
    if [ $attempt -gt $max_attempts ]; then
        echo "Warning: Command failed after $max_attempts attempts: $cmd"
    fi
}

# # CUBES - MAX_SIZE from 3-6
# for MAX_SIZE in 3 4 5 6; do
#     echo "=== Generating cubes with MAX_SIZE=$MAX_SIZE ==="
#     P_TYPE=count MAX_SIZE=$MAX_SIZE python cubes.py 
#     P_TYPE=missing MAX_SIZE=$MAX_SIZE python cubes.py 
#     P_TYPE=surface_area MAX_SIZE=$MAX_SIZE python cubes.py
#     P_TYPE=exposed MAX_SIZE=$MAX_SIZE python cubes.py
#     P_TYPE=colors MAX_SIZE=$MAX_SIZE python cubes.py
#     retry_until_file_exists "P_TYPE=max_color MAX_SIZE=$MAX_SIZE python cubes.py" "cubes_max_color_max${MAX_SIZE}"
#     P_TYPE=project MAX_SIZE=$MAX_SIZE python cubes.py
#     P_TYPE=missing_shape MAX_SIZE=$MAX_SIZE python cubes.py
#     P_TYPE=matching MAX_SIZE=$MAX_SIZE python cubes.py
# done

# # CUBE_PATH - PATH_LENGTH from 5-8
# for PATH_LENGTH in 5 6 7 8; do
#     echo "=== Generating cube_path with PATH_LENGTH=$PATH_LENGTH ==="
#     PATH_LENGTH=$PATH_LENGTH python cube_path.py
# done

# # DICE - NUM_DICE from 3-6
# for NUM_DICE in 3 4 5 6; do
#     echo "=== Generating dice with NUM_DICE=$NUM_DICE ==="
#     P_TYPE=hidden NUM_DICE=$NUM_DICE N_ROLL=5 python dice.py
#     retry_until_file_exists "P_TYPE=max_hidden NUM_DICE=$NUM_DICE N_ROLL=5 python dice.py" "dice_max_hidden_dice${NUM_DICE}"
#     P_TYPE=match NUM_DICE=$NUM_DICE N_ROLL=5 python dice.py
#     P_TYPE=roll NUM_DICE=$NUM_DICE N_ROLL=5 python dice.py
#     P_TYPE=n_roll NUM_DICE=$NUM_DICE N_ROLL=5 python dice.py
#     P_TYPE=fold NUM_DICE=$NUM_DICE N_ROLL=5 python dice.py
# done

# PATH - NUM_SHAPES from 3-6
for NUM_SHAPES in 3 4 5 6; do
    echo "=== Generating path with NUM_SHAPES=$NUM_SHAPES ==="
    P_TYPE=max_dist NUM_SHAPES=$NUM_SHAPES python path.py
    P_TYPE=min_dist NUM_SHAPES=$NUM_SHAPES python path.py
    P_TYPE=order NUM_SHAPES=$NUM_SHAPES python path.py
    P_TYPE=max_time NUM_SHAPES=$NUM_SHAPES python path.py
    P_TYPE=min_time NUM_SHAPES=$NUM_SHAPES python path.py
done

# ROPES - NUM_ROPES from 4-7
# for NUM_ROPES in 4 5 6 7; do
#     echo "=== Generating ropes with NUM_ROPES=$NUM_ROPES ==="
#     P_TYPE=count NUM_ROPES=$NUM_ROPES BENDS_PER_ROPE=3 python ropes.py
#     P_TYPE=cut NUM_ROPES=$NUM_ROPES BENDS_PER_ROPE=3 python ropes.py
#     P_TYPE=closed NUM_ROPES=$NUM_ROPES BENDS_PER_ROPE=3 python ropes.py
#     P_TYPE=order NUM_ROPES=$NUM_ROPES BENDS_PER_ROPE=3 python ropes.py
# done

echo "=== All generations complete ==="
