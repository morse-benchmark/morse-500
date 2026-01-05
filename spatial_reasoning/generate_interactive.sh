#!/usr/bin/env bash
# generate_interactive.sh
# Interactive script to generate SIZE-parameterized videos
# Prompts user for all inputs with helpful guidance

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# ============================================================================
# Welcome Screen
# ============================================================================

clear
echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║                                                                    ║${NC}"
echo -e "${BOLD}${BLUE}║        SIZE-Parameterized Video Generator (Interactive)            ║${NC}"
echo -e "${BOLD}${BLUE}║                                                                    ║${NC}"
echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "This script will generate spatial reasoning videos at your chosen"
echo "difficulty level using the SIZE parameter (0.0 = easy, 1.0 = hard)."
echo ""
echo "Each program will generate videos for ALL its supported problem types."
echo ""

# ============================================================================
# Step 1: Get SIZE parameter
# ============================================================================

echo -e "${BOLD}${CYAN}Step 1: Choose Difficulty Level (SIZE)${NC}"
echo ""
echo "SIZE controls problem difficulty:"
echo "  0.0 - 0.2  → Very Easy (baseline, models should get ~90%+)"
echo "  0.3 - 0.5  → Medium (models start to differentiate)"
echo "  0.6 - 0.8  → Hard (only strong models succeed)"
echo "  0.9 - 1.0  → Very Hard (stress test, expect <50%)"
echo ""

while true; do
    read -p "Enter SIZE value (0.0 to 1.0) [default: 0.5]: " SIZE
    SIZE=${SIZE:-0.5}

    # Validate it's a number
    if [[ ! "$SIZE" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        echo -e "${RED}Error: Please enter a valid number${NC}"
        continue
    fi

    # Check range
    if ! awk -v s="$SIZE" 'BEGIN { exit !(s >= 0.0 && s <= 1.0) }'; then
        echo -e "${YELLOW}Warning: SIZE outside 0.0-1.0 range. It will be clamped.${NC}"
        read -p "Continue anyway? (y/n): " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            continue
        fi
    fi

    break
done

echo -e "${GREEN}✓ SIZE set to: $SIZE${NC}"
echo ""

# ============================================================================
# Step 2: Choose which programs to run
# ============================================================================

echo -e "${BOLD}${CYAN}Step 2: Choose Programs to Run${NC}"
echo ""
echo "Available programs (size is a major factor for these):"
echo "  1) cubes_sized.py        - 3D grid counting and analysis"
echo "  2) ropes_sized.py        - Line intersection and geometry"
echo "  3) domino_count_sized.py - Count dominoes of a specific color"
echo "  4) All programs          - Generate from all 3 programs"
echo ""

read -p "Select option (1-4) [default: 4]: " prog_choice
prog_choice=${prog_choice:-4}

case $prog_choice in
    1) PROGRAMS=("cubes_sized.py");;
    2) PROGRAMS=("ropes_sized.py");;
    3) PROGRAMS=("domino_count_sized.py");;
    4) PROGRAMS=("cubes_sized.py" "ropes_sized.py" "domino_count_sized.py");;
    *)
        echo -e "${RED}Invalid choice. Using all programs.${NC}"
        PROGRAMS=("cubes_sized.py" "ropes_sized.py" "domino_count_sized.py")
        ;;
esac

echo -e "${GREEN}✓ Selected ${#PROGRAMS[@]} program(s)${NC}"
echo ""

# ============================================================================
# Step 3: Number of instances per problem type
# ============================================================================

echo -e "${BOLD}${CYAN}Step 3: Number of Videos per Problem Type${NC}"
echo ""
echo "How many videos should be generated for each problem type?"
echo "(Each program has multiple problem types - see details below)"
echo ""
echo "Problem types by program:"
echo "  cubes_sized:        count, missing, surface_area, exposed, colors, max_color, project (7 types)"
echo "  ropes_sized:        count, cut, closed, order (4 types)"
echo "  domino_count_sized: single problem type (color counting)"
echo ""

read -p "Number of instances per problem type [default: 1]: " NUM_INSTANCES
NUM_INSTANCES=${NUM_INSTANCES:-1}

# Validate it's a positive integer
if ! [[ "$NUM_INSTANCES" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${YELLOW}Invalid number. Using 1 instance.${NC}"
    NUM_INSTANCES=1
fi

echo -e "${GREEN}✓ Will generate $NUM_INSTANCES instance(s) per problem type${NC}"
echo ""

# ============================================================================
# Calculate total videos and confirm
# ============================================================================

# Define problem types for each program using a function
get_problem_types() {
    case "$1" in
        "cubes_sized.py")
            echo "count missing surface_area exposed colors max_color project"
            ;;
        "ropes_sized.py")
            echo "count cut closed order"
            ;;
        "domino_count_sized.py")
            echo "default"
            ;;
    esac
}

# Calculate total
total_videos=0
for program in "${PROGRAMS[@]}"; do
    types=$(get_problem_types "$program")
    num_types=$(echo "$types" | wc -w)
    total_videos=$((total_videos + num_types * NUM_INSTANCES))
done

echo ""
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${BLUE}Summary${NC}"
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  SIZE:              $SIZE"
echo "  Programs:          ${#PROGRAMS[@]} (${PROGRAMS[*]})"
echo "  Instances/type:    $NUM_INSTANCES"
echo "  Total videos:      $total_videos"
echo ""
echo "Breakdown:"
for program in "${PROGRAMS[@]}"; do
    program_name=$(basename "$program" .py)
    types=$(get_problem_types "$program")
    num_types=$(echo "$types" | wc -w)
    prog_total=$((num_types * NUM_INSTANCES))
    echo "  $program_name: $num_types types × $NUM_INSTANCES = $prog_total videos"
done
echo ""
echo "Output will be saved to:"
echo "  Videos:            questions/"
echo "  Solutions:         solutions/"
echo "  Question text:     question_text/"
echo "  Reasoning traces:  reasoning_traces/"
echo ""

read -p "Proceed with generation? (y/n): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${YELLOW}Generation cancelled.${NC}"
    exit 0
fi

# ============================================================================
# Generate videos
# ============================================================================

echo ""
echo -e "${BOLD}${GREEN}Starting generation...${NC}"
echo ""

export SIZE

total_generated=0
total_failed=0
start_time=$(date +%s)

for program in "${PROGRAMS[@]}"; do
    program_name=$(basename "$program" .py)

    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}Program: $program_name${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Get problem types for this program
    types=$(get_problem_types "$program")
    types_array=($types)

    for p_type in "${types_array[@]}"; do
        echo -e "${CYAN}  Problem type: $p_type${NC}"
        export P_TYPE=$p_type

        for instance in $(seq 1 "$NUM_INSTANCES"); do
            echo -n "    Instance ${instance}/${NUM_INSTANCES}: "

            if python "$program" > /dev/null 2>&1; then
                total_generated=$((total_generated + 1))
                echo -e "${GREEN}✓ Success${NC}"
            else
                total_failed=$((total_failed + 1))
                echo -e "${RED}✗ Failed${NC}"
            fi
        done
        echo ""
    done
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))

# ============================================================================
# Final summary
# ============================================================================

echo ""
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}Generation Complete!${NC}"
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Statistics:"
echo "  ✓ Successfully generated: $total_generated videos"
if [ $total_failed -gt 0 ]; then
    echo -e "  ${RED}✗ Failed: $total_failed videos${NC}"
fi
echo "  ⏱ Time elapsed: ${elapsed}s"
echo ""

# Show recently generated files
echo -e "${BOLD}Recent videos (SIZE=$SIZE):${NC}"
ls -lt questions/*size${SIZE}*.mp4 2>/dev/null | head -n 10 | awk '{print "  "$9}' || echo "  (none found)"
echo ""

if [ $total_failed -eq 0 ]; then
    echo -e "${BOLD}${GREEN}🎉 All videos generated successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "  • View videos: open questions/"
    echo "  • Check answers: cat solutions/*size${SIZE}*.txt"
    echo "  • Read reasoning: cat reasoning_traces/*size${SIZE}*.txt"
    echo ""
else
    echo -e "${BOLD}${YELLOW}⚠ Some videos failed. Check error messages above.${NC}"
    echo ""
    echo "Common issues:"
    echo "  • Missing dependencies: pip install -r requirements.txt"
    echo "  • Manim not installed: pip install manim"
    echo ""
fi
