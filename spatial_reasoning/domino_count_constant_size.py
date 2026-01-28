from manim import *
import random
import math
import os
import shutil
from pathlib import Path

# ============================================================================
# COUNT VERSION - Vary Target Count at Fixed Size
# ============================================================================
# This version keeps domino SIZE constant (1% screen coverage per domino)
# and varies how many dominoes are of the target color.
#
# KEY CONCEPT:
# TARGET_COUNT = number of dominoes that are the target color (0 to TOTAL)
#
# Parameters:
# - SIZE_RATIO: Fixed at 0.001 (0.1% screen per domino, ~32x64 pixels)
# - TOTAL_DOMINOES: Fixed total number of dominoes (default 20)
# - TARGET_COUNT: How many are the target color (0 to TOTAL_DOMINOES)
#
# Examples:
# - TARGET_COUNT=1: 1 out of 10 dominoes is target color
# - TARGET_COUNT=5: 5 out of 10 dominoes is target color
# - TARGET_COUNT=10: All 10 dominoes are target color
# ============================================================================

# ============================================================================
# Setup directories for output files
# ============================================================================
Path("questions").mkdir(exist_ok=True)
Path("solutions").mkdir(exist_ok=True)
Path("question_text").mkdir(exist_ok=True)
Path("reasoning_traces").mkdir(exist_ok=True)

# ============================================================================
# Manim configuration
# ============================================================================
config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.preview = False

class DominoCountConstantSize(Scene):
    """
    COUNT version of domino counting.

    Size is fixed at 1% screen coverage per domino.
    TARGET_COUNT controls how many dominoes are the target color.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # ====================================================================
        # FIXED SIZE PARAMETER - 0.1% screen coverage per domino
        # ====================================================================
        self.size_ratio = 0.001  # Fixed at 0.1% (allows more dominoes)

        # ====================================================================
        # TOTAL DOMINOES - Fixed total count
        # ====================================================================
        self.num_dominoes = int(os.getenv("TOTAL_DOMINOES", 20))
        self.num_dominoes = max(1, min(50, self.num_dominoes))

        # ====================================================================
        # TARGET_COUNT PARAMETER - How many are the target color
        # ====================================================================
        self.target_count = int(os.getenv("TARGET_COUNT", 3))
        self.target_count = max(0, min(self.num_dominoes, self.target_count))

        # ====================================================================
        # DOMINO SIZE CALCULATION (same as pixel ratio version)
        # ====================================================================
        screen_pixels = config.pixel_width * config.pixel_height
        target_domino_pixels = self.size_ratio * screen_pixels

        manim_frame_width = config.frame_width
        manim_frame_height = config.frame_height
        pixels_per_unit_x = config.pixel_width / manim_frame_width
        pixels_per_unit_y = config.pixel_height / manim_frame_height

        target_domino_manim_area = target_domino_pixels / (pixels_per_unit_x * pixels_per_unit_y)
        self.domino_width = math.sqrt(target_domino_manim_area / 2)
        self.domino_height = 2.0 * self.domino_width

        # Calculate pixel dimensions for logging
        self.domino_pixel_width = self.domino_width * pixels_per_unit_x
        self.domino_pixel_height = self.domino_height * pixels_per_unit_y

        # ====================================================================
        # COLOR ASSIGNMENT
        # ====================================================================
        self.colors = {
            "red": RED,
            "blue": BLUE,
            "green": GREEN,
            "yellow": YELLOW,
            "purple": PURPLE
        }
        self.color_names = list(self.colors.keys())

        # Randomly select target color
        self.target_color = random.choice(self.color_names)

        # Other colors (excluding target)
        other_colors = [c for c in self.color_names if c != self.target_color]

        # Create color list with exact TARGET_COUNT of target color
        self.domino_colors = []

        # Add target color dominoes
        for _ in range(self.target_count):
            self.domino_colors.append(self.target_color)

        # Add other colors for remaining dominoes
        for _ in range(self.num_dominoes - self.target_count):
            self.domino_colors.append(random.choice(other_colors))

        # Shuffle to randomize positions
        random.shuffle(self.domino_colors)

        # The answer is exactly TARGET_COUNT
        self.answer = self.target_count

        # Reasoning trace storage
        self.reasoning_trace = []
        self.scene_events = []

    def log_event(self, description):
        """Log a scene event with video timestamp."""
        current_time = self.renderer.time
        self.scene_events.append({
            'time': current_time,
            'description': description
        })

    def format_time(self, seconds):
        """Format seconds as M:SS for display."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def construct(self):
        """Main scene construction."""

        # Gradient background
        bg = Rectangle(
            height=config.frame_height,
            width=config.frame_width
        ).set_color(
            color_gradient([random_bright_color(), random_bright_color()], 5)
        ).set_opacity(0.6).set_z_index(-2)
        self.add(bg)

        self.log_event(f"Scene starts - TARGET_COUNT: {self.target_count} out of {self.num_dominoes}")
        self.log_event(f"Fixed size: {self.size_ratio*100:.1f}% per domino ({self.domino_pixel_width:.0f}x{self.domino_pixel_height:.0f} px)")

        # Show instruction title
        title = Text(f"Count the {self.target_color.upper()} dominoes", font_size=48, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.log_event(f"Instruction appears: count {self.target_color} dominoes")
        self.wait(2)
        self.play(FadeOut(title))

        # Create dominoes - arrange in grid if they don't fit in one row
        dominoes = VGroup()
        spacing = max(0.05, 0.1 * self.domino_width)

        # Calculate how many dominoes fit per row
        usable_width = config.frame_width * 0.9
        domino_with_spacing = self.domino_width + spacing
        dominoes_per_row = max(1, int(usable_width / domino_with_spacing))

        # Calculate number of rows needed
        num_rows = math.ceil(self.num_dominoes / dominoes_per_row)

        # Row height
        row_height = self.domino_height + spacing

        for i in range(self.num_dominoes):
            color_name = self.domino_colors[i]
            domino = Rectangle(
                width=self.domino_width,
                height=self.domino_height,
                fill_color=self.colors[color_name],
                fill_opacity=1,
                stroke_color=WHITE,
                stroke_width=max(1, min(3, self.domino_width * 5))
            )

            # Calculate row and column
            row = i // dominoes_per_row
            col = i % dominoes_per_row
            dominoes_in_this_row = min(dominoes_per_row, self.num_dominoes - row * dominoes_per_row)

            # Calculate position
            row_width = dominoes_in_this_row * domino_with_spacing - spacing
            start_x = -row_width / 2
            x_pos = start_x + col * domino_with_spacing + self.domino_width / 2

            # Vertical position (center rows vertically)
            total_height = num_rows * row_height - spacing
            start_y = total_height / 2 - self.domino_height / 2
            y_pos = start_y - row * row_height

            domino.move_to([x_pos, y_pos, 0])

            # Initial rotation (standing)
            domino.rotate(-10 * DEGREES)

            dominoes.add(domino)

            self.log_event(f"Domino {i+1} appears ({color_name})")

        # Show dominoes appearing
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dominoes], lag_ratio=0.1))
        self.wait(1)
        self.log_event(f"All {self.num_dominoes} dominoes visible")

        # Domino falling animation
        self.log_event("Domino chain reaction begins")
        for i, domino in enumerate(dominoes):
            self.play(
                Rotate(domino, angle=-80 * DEGREES, about_point=domino.get_bottom()),
                run_time=0.2
            )
        self.log_event("Domino chain reaction complete")

        self.wait(3)

        # Build reasoning trace
        self.build_reasoning_trace()

        # Save outputs
        base_name = f"domino_count_c{self.target_count}_t{self.num_dominoes}_seed{self.seed}"

        # Use custom output directory if specified
        output_dir = os.getenv("OUTPUT_DIR", "questions")
        Path(output_dir).mkdir(exist_ok=True)

        with open(f"{output_dir}/{base_name}.txt", "w") as f:
            f.write(str(self.answer))

        with open(f"question_text/{base_name}.txt", "w") as f:
            f.write(f"Observe the falling dominoes. How many {self.target_color} dominoes fell? Answer with a single integer.")

        with open(f"reasoning_traces/{base_name}.txt", "w") as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self):
        """Build comprehensive reasoning trace."""
        self.reasoning_trace.append(f"**Question:** How many {self.target_color} dominoes fell?")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"**TARGET_COUNT Parameter:** {self.target_count}")
        self.reasoning_trace.append(f"**Total dominoes:** {self.num_dominoes}")
        self.reasoning_trace.append(f"**Fixed size:** {self.size_ratio*100:.1f}% screen per domino")
        self.reasoning_trace.append(f"**Domino size (pixels):** {self.domino_pixel_width:.0f} x {self.domino_pixel_height:.0f}")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("### Scene Timeline")
        self.reasoning_trace.append("")

        for event in self.scene_events:
            time_str = self.format_time(event['time'])
            self.reasoning_trace.append(f"At {time_str}: {event['description']}")

        self.reasoning_trace.append("")
        self.reasoning_trace.append("### Color Distribution")
        self.reasoning_trace.append("")

        color_counts = {color: self.domino_colors.count(color) for color in self.color_names}
        for color, count in sorted(color_counts.items(), key=lambda x: -x[1]):
            marker = " <-- TARGET" if color == self.target_color else ""
            self.reasoning_trace.append(f"- {color.capitalize()}: {count} dominoes{marker}")

        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"### Answer: {self.answer} {self.target_color} dominoes")
        self.reasoning_trace.append(f"\\boxed{{{self.answer}}}")

# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    scene = DominoCountConstantSize()
    scene.render()

    output = Path("manim_output/videos/1080p30/DominoCountConstantSize.mp4")
    output_dir = os.getenv("OUTPUT_DIR", "questions")
    Path(output_dir).mkdir(exist_ok=True)

    if output.exists():
        filename = f"domino_count_c{scene.target_count}_t{scene.num_dominoes}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"{output_dir}/{filename}")
        print(f"Video saved: {output_dir}/{filename}")
        print(f"TARGET_COUNT: {scene.target_count} {scene.target_color} dominoes out of {scene.num_dominoes} total")
        print(f"Fixed size: {scene.size_ratio*100:.1f}% per domino ({scene.domino_pixel_width:.0f}x{scene.domino_pixel_height:.0f} px)")
        print(f"Answer: {scene.answer}")
    else:
        print("Error: Expected output file not found")

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
        print("Cleanup complete")
