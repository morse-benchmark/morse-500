#!/bin/bash

# List of files to generate
FILES="
dice_max_hidden_dice5_seed9779.txt
"

# Loop through each filename
for FILE in $FILES; do
    # Extract SEED (numbers after 'seed' and before .txt)
    SEED=$(echo "$FILE" | grep -o 'seed[0-9]*' | sed 's/seed//')
    
    # === CUBE PATH LOGIC ===
    if [[ "$FILE" == cube_path* ]]; then
        # Extract Length (from 'len5' etc)
        LEN=$(echo "$FILE" | grep -o 'len[0-9]*' | sed 's/len//')
        
        echo "Running: CUBE PATH | Len: $LEN | Seed: $SEED"
        PATH_LENGTH=$LEN SEED=$SEED python cube_path.py

    # === CUBES LOGIC ===
    elif [[ "$FILE" == cubes_* ]]; then
        # Extract Max Size (from 'max3' etc)
        SIZE=$(echo "$FILE" | grep -o 'max[0-9]*' | sed 's/max//')
        # Extract Type (remove cubes_ prefix and _max... suffix)
        PTYPE=$(echo "$FILE" | sed 's/^cubes_//' | sed 's/_max.*//')
        
        echo "Running: CUBES | Type: $PTYPE | Size: $SIZE | Seed: $SEED"
        P_TYPE=$PTYPE MAX_SIZE=$SIZE SEED=$SEED python cubes.py

    # === DICE LOGIC ===
    elif [[ "$FILE" == dice_* ]]; then
        # Extract Num Dice (from 'dice3' etc)
        # Note: filenames like dice_match_dice5... have two "dice", we want the one with number
        NDICE=$(echo "$FILE" | grep -o 'dice[0-9]*' | sed 's/dice//')
        # Extract Type (remove dice_ prefix and _dice... suffix)
        PTYPE=$(echo "$FILE" | sed 's/^dice_//' | sed 's/_dice[0-9].*//')
        
        echo "Running: DICE | Type: $PTYPE | Num: $NDICE | Seed: $SEED"
        P_TYPE=$PTYPE NUM_DICE=$NDICE N_ROLL=5 SEED=$SEED python dice.py

    # === PATH LOGIC ===
    elif [[ "$FILE" == path_* ]]; then
        # Extract Shapes (from 'shapes3' etc)
        NSHAPES=$(echo "$FILE" | grep -o 'shapes[0-9]*' | sed 's/shapes//')
        # Extract Type (remove path_ prefix and _shapes... suffix)
        PTYPE=$(echo "$FILE" | sed 's/^path_//' | sed 's/_shapes.*//')
        
        echo "Running: PATH | Type: $PTYPE | Shapes: $NSHAPES | Seed: $SEED"
        P_TYPE=$PTYPE NUM_SHAPES=$NSHAPES SEED=$SEED python path.py

    # === ROPES LOGIC ===
    elif [[ "$FILE" == ropes_* ]]; then
        # Extract Num Ropes (from 'ropes4' etc)
        NROPES=$(echo "$FILE" | grep -o 'ropes[0-9]*' | sed 's/ropes//')
        # Extract Type (remove ropes_ prefix and _ropes... suffix)
        PTYPE=$(echo "$FILE" | sed 's/^ropes_//' | sed 's/_ropes.*//')
        
        echo "Running: ROPES | Type: $PTYPE | Ropes: $NROPES | Seed: $SEED"
        P_TYPE=$PTYPE NUM_ROPES=$NROPES BENDS_PER_ROPE=3 SEED=$SEED python ropes.py
    
    else
        echo "Skipping unrecognized file: $FILE"
    fi
done