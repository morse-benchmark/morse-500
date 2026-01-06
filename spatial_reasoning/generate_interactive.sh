#!/usr/bin/env bash
# generate_interactive.sh
# Interactive script to generate parameterized videos
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
echo -e "${BOLD}${BLUE}║        Parameterized Video Generator (Interactive)                 ║${NC}"
echo -e "${BOLD}${BLUE}║                                                                    ║${NC}"
echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "This script will generate spatial reasoning videos with parameterized"
echo "difficulty control. Choose between SIZE or DENSITY metrics."
echo ""

# ============================================================================
# Step 0: Choose Metric Type
# ============================================================================

echo -e "${BOLD}${CYAN}Step 0: Choose Difficulty Metric${NC}"
echo ""
echo "Select which metric to use for difficulty control:"
echo ""
echo "  1) SIZE      - Controls object size/count (fewer/smaller → more/larger)"
echo "                 Examples: cube grid size, number of dominoes"
echo ""
echo "  2) DENSITY   - Controls clutter/density (sparse → dense)"
echo "                 Examples: cube grid density, rope intersection density"
echo ""
echo "  3) FREQUENCY - Controls visual complexity/pattern detail (coarse → fine)"
echo "                 Examples: pattern granularity, wave frequency, spacing patterns"
echo ""

read -p "Select metric (1-3) [default: 1]: " metric_choice
metric_choice=${metric_choice:-1}

case $metric_choice in
    1)
        METRIC="SIZE"
        METRIC_DESC="SIZE controls object size and count"
        ;;
    2)
        METRIC="DENSITY"
        METRIC_DESC="DENSITY controls visual clutter and object density"
        ;;
    3)
        METRIC="FREQUENCY"
        METRIC_DESC="FREQUENCY controls visual complexity and pattern detail"
        ;;
    *)
        echo -e "${YELLOW}Invalid choice. Using SIZE.${NC}"
        METRIC="SIZE"
        METRIC_DESC="SIZE controls object size and count"
        ;;
esac

echo -e "${GREEN}✓ Selected metric: $METRIC${NC}"
echo ""

# ============================================================================
# Step 1: Get parameter value
# ============================================================================

echo -e "${BOLD}${CYAN}Step 1: Choose Difficulty Level ($METRIC)${NC}"
echo ""
if [ "$METRIC" = "SIZE" ]; then
    echo "$METRIC controls problem difficulty:"
    echo "  0.0 - 0.2  → Very Easy (baseline, models should get ~90%+)"
    echo "  0.3 - 0.5  → Medium (models start to differentiate)"
    echo "  0.6 - 0.8  → Hard (only strong models succeed)"
    echo "  0.9 - 1.0  → Very Hard (stress test, expect <50%)"
elif [ "$METRIC" = "DENSITY" ]; then
    echo "$METRIC controls visual clutter:"
    echo "  0.0 - 0.2  → Very Sparse (minimal objects, easy to track)"
    echo "  0.3 - 0.5  → Medium Density (moderate clutter)"
    echo "  0.6 - 0.8  → High Density (crowded, requires focus)"
    echo "  0.9 - 1.0  → Very Dense (maximum clutter, very challenging)"
else
    echo "$METRIC controls visual complexity:"
    echo "  0.0 - 0.2  → Coarse (simple patterns, easy to perceive)"
    echo "  0.3 - 0.5  → Medium (standard pattern granularity)"
    echo "  0.6 - 0.8  → Fine (complex patterns, requires attention)"
    echo "  0.9 - 1.0  → Very Fine (intricate detail, very challenging)"
fi
echo ""

while true; do
    read -p "Enter $METRIC value (0.0 to 1.0) [default: 0.5]: " PARAM_VALUE
    PARAM_VALUE=${PARAM_VALUE:-0.5}

    # Validate it's a number
    if [[ ! "$PARAM_VALUE" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        echo -e "${RED}Error: Please enter a valid number${NC}"
        continue
    fi

    # Check range
    if ! awk -v s="$PARAM_VALUE" 'BEGIN { exit !(s >= 0.0 && s <= 1.0) }'; then
        echo -e "${YELLOW}Warning: $METRIC outside 0.0-1.0 range. It will be clamped.${NC}"
        read -p "Continue anyway? (y/n): " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            continue
        fi
    fi

    break
done

# Set the appropriate environment variable based on metric choice
if [ "$METRIC" = "SIZE" ]; then
    export SIZE=$PARAM_VALUE
    echo -e "${GREEN}✓ SIZE set to: $SIZE${NC}"
elif [ "$METRIC" = "DENSITY" ]; then
    export DENSITY=$PARAM_VALUE
    echo -e "${GREEN}✓ DENSITY set to: $DENSITY${NC}"
else
    export FREQUENCY=$PARAM_VALUE
    echo -e "${GREEN}✓ FREQUENCY set to: $FREQUENCY${NC}"
fi
echo ""

# ============================================================================
# Step 2: Choose which programs to run
# ============================================================================

echo -e "${BOLD}${CYAN}Step 2: Choose Programs to Run${NC}"
echo ""

if [ "$METRIC" = "SIZE" ]; then
    echo "Available programs for SIZE metric:"
    echo "  1) cubes_sized.py        - 3D grid counting and analysis"
    echo "  2) ropes_sized.py        - Line intersection and geometry"
    echo "  3) domino_count_sized.py - Count dominoes of a specific color"
    echo "  4) All SIZE programs     - Generate from all 3 programs"
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
elif [ "$METRIC" = "DENSITY" ]; then
    echo "Available programs for DENSITY metric:"
    echo "  1) cubes_density.py        - 3D grid with varying cube density"
    echo "  2) ropes_density.py        - Line intersections with varying density"
    echo "  3) domino_count_density.py - Domino counting with clutter control"
    echo "  4) All DENSITY programs    - Generate from all 3 programs"
    echo ""

    read -p "Select option (1-4) [default: 4]: " prog_choice
    prog_choice=${prog_choice:-4}

    case $prog_choice in
        1) PROGRAMS=("cubes_density.py");;
        2) PROGRAMS=("ropes_density.py");;
        3) PROGRAMS=("domino_count_density.py");;
        4) PROGRAMS=("cubes_density.py" "ropes_density.py" "domino_count_density.py");;
        *)
            echo -e "${RED}Invalid choice. Using all programs.${NC}"
            PROGRAMS=("cubes_density.py" "ropes_density.py" "domino_count_density.py")
            ;;
    esac
else
    echo "Available programs for FREQUENCY metric:"
    echo "  1) cubes_frequency.py        - 3D grid with pattern granularity control"
    echo "  2) ropes_frequency.py        - Line intersections with wave frequency"
    echo "  3) domino_count_frequency.py - Domino counting with spacing patterns"
    echo "  4) All FREQUENCY programs    - Generate from all 3 programs"
    echo ""

    read -p "Select option (1-4) [default: 4]: " prog_choice
    prog_choice=${prog_choice:-4}

    case $prog_choice in
        1) PROGRAMS=("cubes_frequency.py");;
        2) PROGRAMS=("ropes_frequency.py");;
        3) PROGRAMS=("domino_count_frequency.py");;
        4) PROGRAMS=("cubes_frequency.py" "ropes_frequency.py" "domino_count_frequency.py");;
        *)
            echo -e "${RED}Invalid choice. Using all programs.${NC}"
            PROGRAMS=("cubes_frequency.py" "ropes_frequency.py" "domino_count_frequency.py")
            ;;
    esac
fi

echo -e "${GREEN}✓ Selected ${#PROGRAMS[@]} program(s)${NC}"
echo ""

# Check if all selected programs exist
echo "Checking if programs exist..."
missing_programs=()
for program in "${PROGRAMS[@]}"; do
    if [ ! -f "$program" ]; then
        missing_programs+=("$program")
    fi
done

if [ ${#missing_programs[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}ERROR: The following programs do not exist:${NC}"
    for prog in "${missing_programs[@]}"; do
        echo -e "  ${RED}✗ $prog${NC}"
    done
    echo ""
    if [ "$METRIC" = "DENSITY" ]; then
        echo -e "${YELLOW}Note: DENSITY programs haven't been created yet.${NC}"
        echo -e "${YELLOW}Please choose option 1 (SIZE) instead, or create the DENSITY programs first.${NC}"
        echo ""
        echo "To create DENSITY programs:"
        echo "  1. cp cubes_sized.py cubes_density.py"
        echo "  2. cp ropes_sized.py ropes_density.py"
        echo "  3. cp domino_count_sized.py domino_count_density.py"
        echo "  4. Edit each file to use DENSITY parameter (see DENSITY_METRIC_GUIDE.md)"
    fi
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ All programs exist${NC}"
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
        "cubes_sized.py"|"cubes_density.py"|"cubes_frequency.py")
            echo "count missing surface_area exposed colors max_color project"
            ;;
        "ropes_sized.py"|"ropes_density.py"|"ropes_frequency.py")
            echo "count cut closed order"
            ;;
        "domino_count_sized.py"|"domino_count_density.py"|"domino_count_frequency.py")
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
echo "  Metric:            $METRIC"
if [ "$METRIC" = "SIZE" ]; then
    echo "  SIZE:              $SIZE"
elif [ "$METRIC" = "DENSITY" ]; then
    echo "  DENSITY:           $DENSITY"
else
    echo "  FREQUENCY:         $FREQUENCY"
fi
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

# Export the appropriate parameter
if [ "$METRIC" = "SIZE" ]; then
    export SIZE
elif [ "$METRIC" = "DENSITY" ]; then
    export DENSITY
else
    export FREQUENCY
fi

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
            echo -e "    Instance ${instance}/${NUM_INSTANCES}:"
            echo -e "${YELLOW}    ───────────────────────────────────────────────${NC}"

            # Run with full error output visible
            if /Users/ankitnakhawa/miniconda3/envs/morse-500/bin/python "$program" 2>&1; then
                total_generated=$((total_generated + 1))
                echo -e "${YELLOW}    ───────────────────────────────────────────────${NC}"
                echo -e "    ${GREEN}✓ Success${NC}"
            else
                total_failed=$((total_failed + 1))
                echo -e "${YELLOW}    ───────────────────────────────────────────────${NC}"
                echo -e "    ${RED}✗ Failed (see error above)${NC}"
            fi
            echo ""
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
if [ "$METRIC" = "SIZE" ]; then
    echo -e "${BOLD}Recent videos (SIZE=$SIZE):${NC}"
    ls -lt questions/*size${SIZE}*.mp4 2>/dev/null | head -n 10 | awk '{print "  "$9}' || echo "  (none found)"
elif [ "$METRIC" = "DENSITY" ]; then
    echo -e "${BOLD}Recent videos (DENSITY=$DENSITY):${NC}"
    ls -lt questions/*density${DENSITY}*.mp4 2>/dev/null | head -n 10 | awk '{print "  "$9}' || echo "  (none found)"
else
    echo -e "${BOLD}Recent videos (FREQUENCY=$FREQUENCY):${NC}"
    ls -lt questions/*frequency${FREQUENCY}*.mp4 2>/dev/null | head -n 10 | awk '{print "  "$9}' || echo "  (none found)"
fi
echo ""

if [ $total_failed -eq 0 ]; then
    echo -e "${BOLD}${GREEN}🎉 All videos generated successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "  • View videos: open questions/"
    if [ "$METRIC" = "SIZE" ]; then
        echo "  • Check answers: cat solutions/*size${SIZE}*.txt"
        echo "  • Read reasoning: cat reasoning_traces/*size${SIZE}*.txt"
    elif [ "$METRIC" = "DENSITY" ]; then
        echo "  • Check answers: cat solutions/*density${DENSITY}*.txt"
        echo "  • Read reasoning: cat reasoning_traces/*density${DENSITY}*.txt"
    else
        echo "  • Check answers: cat solutions/*frequency${FREQUENCY}*.txt"
        echo "  • Read reasoning: cat reasoning_traces/*frequency${FREQUENCY}*.txt"
    fi
    echo ""
else
    echo -e "${BOLD}${YELLOW}⚠ Some videos failed. Check error messages above.${NC}"
    echo ""
    echo "Common issues:"
    echo "  • Missing dependencies: pip install -r requirements.txt"
    echo "  • Manim not installed: pip install manim"
    echo ""
fi
