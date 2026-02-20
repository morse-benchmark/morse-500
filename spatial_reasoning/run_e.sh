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

for ITER in 5; do # 1 2 3 4; do
    echo "================ Iteration $ITER ================="

    for NUM_DICE in 5; do
        echo "=== Generating dice with NUM_DICE=$NUM_DICE ==="
        P_TYPE=hidden NUM_DICE=$NUM_DICE N_ROLL=5 python dice.py
        retry_until_file_exists "P_TYPE=max_hidden NUM_DICE=$NUM_DICE N_ROLL=5 python dice.py" "dice_max_hidden_dice${NUM_DICE}"
    done

done


echo "=== All generations complete ==="
