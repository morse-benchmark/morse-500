
# size range 5 - 15
python maze_pixel_distance_gap.py  --size 10
python maze_pixel_largest_distance.py --size 10
python maze_plot_endpoint.py  --size 10
python maze_plot_farthest_moves.py  --size 10
python maze_plot_visit_all_letters.py --size 10
python maze_plot_endpoint.py  --size 10 


python frozen_lake_holes.py --size 10 --question_name "count"
python frozen_lake_holes.py --size 10 --n_neighbors 4 --question_name "connected_count"
python frozen_lake_holes.py --size 10 --n_neighbors 8 --question_name "connected_count"
python frozen_lake_holes.py --size 10 --n_neighbors 4 --question_name "connected_area"
python frozen_lake_holes.py --size 10 --n_neighbors 8 --question_name "connected_area"


python frozen_lake_paths.py --size 10 --question_name "count"
python frozen_lake_paths.py --size 10 --question_name "min_length"
python frozen_lake_paths.py --size 10 --question_name "agent_steps"
python frozen_lake_correct_sequence.py --size 10


# increasing difficulty
python frozen_lake_correct_sequence.py --size 5 --n_options 4
python frozen_lake_correct_sequence.py --size 10 --n_options 6
python frozen_lake_correct_sequence.py --size 10 --n_options 6 --use_fog --visibility_range 4
python frozen_lake_correct_sequence.py --size 15 --n_options 6 --use_fog --visibility_range 2
