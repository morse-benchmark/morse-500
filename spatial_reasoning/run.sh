#!/bin/bash


# MAX_SIZE can be 3, 4, 5+
P_TYPE=count MAX_SIZE=3 python cubes.py 
P_TYPE=missing MAX_SIZE=3 python cubes.py 
P_TYPE=surface_area MAX_SIZE=3 python cubes.py
P_TYPE=exposed MAX_SIZE=3 python cubes.py
P_TYPE=colors MAX_SIZE=3 python cubes.py
P_TYPE=max_color MAX_SIZE=3 python cubes.py
P_TYPE=project MAX_SIZE=3 python cubes.py
P_TYPE=missing_shape MAX_SIZE=3 python cubes.py
P_TYPE=matching MAX_SIZE=3 python cubes.py


PATH_LENGTH=5 python cube_path.py


P_TYPE=hidden NUM_DICE=4 N_ROLL=5 python dice.py
P_TYPE=max_hidden NUM_DICE=4 N_ROLL=5 python dice.py
P_TYPE=match NUM_DICE=4 N_ROLL=5 python dice.py
P_TYPE=roll NUM_DICE=4 N_ROLL=5 python dice.py
P_TYPE=n_roll NUM_DICE=4 N_ROLL=5 python dice.py
P_TYPE=fold NUM_DICE=4 N_ROLL=5 python dice.py


# NUM_SHAPES can be 3, 4, 5, 6 +
P_TYPE=max_dist NUM_SHAPES=3 python path.py
P_TYPE=min_dist NUM_SHAPES=3 python path.py
P_TYPE=order NUM_SHAPES=3 python path.py
P_TYPE=max_time NUM_SHAPES=3 python path.py
P_TYPE=min_time NUM_SHAPES=3 python path.py


P_TYPE=count NUM_ROPES=4 BENDS_PER_ROPE=3 python ropes.py
P_TYPE=cut NUM_ROPES=4 BENDS_PER_ROPE=3 python ropes.py
P_TYPE=closed NUM_ROPES=4 BENDS_PER_ROPE=3 python ropes.py
P_TYPE=order NUM_ROPES=4 BENDS_PER_ROPE=3


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
