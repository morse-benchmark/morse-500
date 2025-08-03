
# need to have the AnimatedKnots dir
# https://drive.google.com/file/d/1W2AGCty0JaltTnYu9Os0-fhxp1psj2Rs/view?usp=sharing

# Run specific tasks with default difficulties
python make_questions_rope_knot.py --tasks knot_reorder --difficulties easy --max-knots 10
python make_questions_rope_knot.py --tasks knot_remaining_order --difficulties medium --max-knots 10
python make_questions_rope_knot.py --tasks knot_remaining_count --difficulties hard --max-knots 10
python make_questions_rope_knot.py --tasks knot_reorder_effect --difficulties easy --max-knots 10 --with-effects

# # Process only first 10 knots with custom slide duration
# python make_questions_rope_knot.py --tasks knot_reorder knot_remaining_order --max-knots 10 --slide-duration 2.0