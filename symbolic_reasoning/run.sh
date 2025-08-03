#!/bin/bash

NUM_SHUFFLES=3 python anagram_all_distance.py # number of shuffles. 3, 5, 7, 9
NUM_SHUFFLES=3 python anagram_partial_distance.py # number of shuffles. 3, 5, 7, 9
NUM_SHUFFLES=3 python anagram_position.py # number of shuffles. 3, 5, 7, 9
NUM_SHUFFLES=3 python anagram_num_shuffles.py # number of shuffles. 3, 5, 7, 9


# REPEAT_TIMES=1
# for shuffles in 3 5 7 9; do
#     for i in $(seq 1 $REPEAT_TIMES); do
#         NUM_SHUFFLES=$shuffles python anagram_all_distance.py
#     done
# done
