from manim import *
import random
import math
import os
import shutil
from pathlib import Path

# ============================================================================
# PIXEL RATIO VERSION - Single Domino Screen Coverage
# ============================================================================
# This version uses SIZE_RATIO to control the percentage of screen pixels
# occupied by a SINGLE domino.
#
# KEY CONCEPT:
# SIZE_RATIO = (single domino area in pixels) / (total screen pixels)
#
# Examples:
# - At 0.00001 (0.001%): Each domino is tiny (~20 pixels)
# - At 0.0001 (0.01%): Each domino is very small (~207 pixels)
# - At 0.001 (0.1%): Each domino is small (~2,074 pixels)
# - At 0.01 (1%): Each domino is medium (~20,736 pixels, ~102x204)
# - At 0.02 (2%): Each domino is larger (~41,472 pixels, ~144x288)
# - At 0.05 (5%): Each domino is large (~103,680 pixels, ~228x456)
#
# Screen: 1920x1080 = 2,073,600 pixels
#
# PRESETS (use PRESET=1,2,3,4 to select):
# - PRESET=1: 10 total dominoes, answer = 3
# - PRESET=2: 15 total dominoes, answer = 5
# - PRESET=3: 20 total dominoes, answer = 7
# - PRESET=4: 25 total dominoes, answer = 9
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

class DominoPixelRatio(Scene):
    """
    PIXEL RATIO version of domino counting.

    SIZE_RATIO controls the percentage of screen area occupied by ONE domino.
    Number of dominoes is fixed at 10.

    Formula:
    SIZE_RATIO = (single_domino_pixels) / (screen_pixels)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # ====================================================================
        # SIZE_RATIO PARAMETER - Single Domino Area as % of Screen (0.0 to 1.0)
        # ====================================================================
        self.size_ratio = float(os.getenv("SIZE_RATIO", 0.01))  # Default 1%

        # Clamp to valid range
        # Below 0.0001% (0.000001): domino would be < 2 pixels
        # Above 10% (0.1): domino would be huge
        self.size_ratio = max(0.000001, min(0.1, self.size_ratio))

        # ====================================================================
        # PRESET PARAMETER - Predefined total/answer combinations
        # ====================================================================
        # PRESET=1: 10 total, answer=3
        # PRESET=2: 15 total, answer=5
        # PRESET=3: 20 total, answer=7
        # PRESET=4: 25 total, answer=9
        PRESETS = {
            1: (10, 3),
            2: (15, 5),
            3: (20, 7),
            4: (25, 9),
        }

        preset = os.getenv("PRESET")
        if preset and int(preset) in PRESETS:
            self.num_dominoes, self.target_count = PRESETS[int(preset)]
            self.use_preset = True
        else:
            # Default: 10 dominoes, random answer
            self.num_dominoes = 10
            self.target_count = None  # Will be determined randomly
            self.use_preset = False

        # ====================================================================
        # SINGLE DOMINO PIXEL COVERAGE CALCULATION
        # ====================================================================
        # SIZE_RATIO = (single domino area) / (screen area)
        # single_domino_area = SIZE_RATIO × screen_area

        # Screen dimensions in pixels
        screen_pixels = config.pixel_width * config.pixel_height  # 2,073,600

        # Target area for ONE domino in pixels
        target_domino_pixels = self.size_ratio * screen_pixels

        # Manim coordinate system conversion
        # frame_width (~14.22) maps to pixel_width (1920)
        # So: pixels_per_manim_unit = 1920 / 14.22 ≈ 135
        manim_frame_width = config.frame_width
        manim_frame_height = config.frame_height
        pixels_per_unit_x = config.pixel_width / manim_frame_width
        pixels_per_unit_y = config.pixel_height / manim_frame_height

        # Target area in Manim units squared
        # pixel_area = manim_area × pixels_per_unit_x × pixels_per_unit_y
        # manim_area = pixel_area / (pixels_per_unit_x × pixels_per_unit_y)
        target_domino_manim_area = target_domino_pixels / (pixels_per_unit_x * pixels_per_unit_y)

        # Calculate domino dimensions (height = 2 × width for standard aspect ratio)
        # area = width × height = width × (2 × width) = 2 × width²
        # width = sqrt(area / 2)
        self.domino_width = math.sqrt(target_domino_manim_area / 2)

        # Ensure minimum visible size (at least 2 pixels wide)
        min_width = 2 / pixels_per_unit_x
        self.domino_width = max(min_width, self.domino_width)

        # Height is 2× width (standard domino aspect ratio)
        self.domino_height = 2.0 * self.domino_width

        # Calculate actual coverage achieved (for verification)
        actual_domino_manim_area = self.domino_width * self.domino_height
        actual_domino_pixels = actual_domino_manim_area * pixels_per_unit_x * pixels_per_unit_y
        self.actual_coverage = actual_domino_pixels / screen_pixels
        self.domino_pixel_width = self.domino_width * pixels_per_unit_x
        self.domino_pixel_height = self.domino_height * pixels_per_unit_y

        # Define colors
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

        # Assign colors to dominoes
        if self.use_preset and self.target_count is not None:
            # Preset mode: guarantee exact target_count of target color
            other_colors = [c for c in self.color_names if c != self.target_color]
            self.domino_colors = []
            # Add target color dominoes
            for _ in range(self.target_count):
                self.domino_colors.append(self.target_color)
            # Add other colors for remaining dominoes
            for _ in range(self.num_dominoes - self.target_count):
                self.domino_colors.append(random.choice(other_colors))
            # Shuffle to randomize positions
            random.shuffle(self.domino_colors)
            self.answer = self.target_count
        else:
            # Random mode: assign random colors
            self.domino_colors = [random.choice(self.color_names) for _ in range(self.num_dominoes)]
            self.answer = self.domino_colors.count(self.target_color)

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

        self.log_event(f"Scene starts - SIZE_RATIO: {self.size_ratio:.6f} ({self.size_ratio*100:.4f}% single domino coverage)")
        self.log_event(f"Actual single domino coverage: {self.actual_coverage*100:.4f}% of screen")
        self.log_event(f"Domino size: {self.domino_pixel_width:.1f} x {self.domino_pixel_height:.1f} pixels")
        self.log_event(f"{self.num_dominoes} dominoes total")

        # Show instruction title
        title = Text(f"Count the {self.target_color.upper()} dominoes", font_size=48, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.log_event(f"Instruction appears: count {self.target_color} dominoes")
        self.wait(2)
        self.play(FadeOut(title))

        # Create dominoes - arrange in grid if they don't fit in one row
        dominoes = VGroup()
        spacing = max(0.05, 0.1 * self.domino_width)  # Proportional spacing, min 0.05 units

        # Calculate how many dominoes fit per row
        usable_width = config.frame_width * 0.9  # Leave 10% margin
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
                stroke_width=max(1, min(3, self.domino_width * 5))  # Scale stroke with size
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
        if self.use_preset:
            preset_num = [k for k, v in {1: (10, 3), 2: (15, 5), 3: (20, 7), 4: (25, 9)}.items()
                         if v == (self.num_dominoes, self.target_count)][0]
            base_name = f"domino_pixelratio_p{preset_num}_r{self.size_ratio:.6f}_seed{self.seed}"
        else:
            base_name = f"domino_pixelratio_r{self.size_ratio:.6f}_seed{self.seed}"

        # Use custom output directory if specified, otherwise use questions/
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
        self.reasoning_trace.append(f"**SIZE_RATIO Parameter:** {self.size_ratio:.6f} ({self.size_ratio*100:.4f}% single domino screen coverage)")
        self.reasoning_trace.append(f"**Actual single domino coverage:** {self.actual_coverage*100:.4f}% of screen")
        self.reasoning_trace.append(f"**Number of dominoes:** {self.num_dominoes} (fixed)")
        self.reasoning_trace.append(f"**Domino size (Manim):** {self.domino_width:.4f} x {self.domino_height:.4f} units")
        self.reasoning_trace.append(f"**Domino size (pixels):** {self.domino_pixel_width:.1f} x {self.domino_pixel_height:.1f} pixels")
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
            self.reasoning_trace.append(f"- {color.capitalize()}: {count} dominoes")

        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"### Answer: {self.answer} {self.target_color} dominoes")
        self.reasoning_trace.append(f"\\boxed{{{self.answer}}}")

# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    scene = DominoPixelRatio()
    scene.render()

    output = Path("manim_output/videos/1080p30/DominoPixelRatio.mp4")
    output_dir = os.getenv("OUTPUT_DIR", "questions")
    Path(output_dir).mkdir(exist_ok=True)

    if output.exists():
        if scene.use_preset:
            preset_num = [k for k, v in {1: (10, 3), 2: (15, 5), 3: (20, 7), 4: (25, 9)}.items()
                         if v == (scene.num_dominoes, scene.target_count)][0]
            filename = f"domino_pixelratio_p{preset_num}_r{scene.size_ratio:.6f}_seed{scene.seed}.mp4"
        else:
            filename = f"domino_pixelratio_r{scene.size_ratio:.6f}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"{output_dir}/{filename}")
        print(f"Video saved: {output_dir}/{filename}")
        if scene.use_preset:
            print(f"PRESET: {preset_num} (total={scene.num_dominoes}, answer={scene.target_count})")
        print(f"SIZE_RATIO: {scene.size_ratio:.6f} ({scene.size_ratio*100:.4f}% single domino coverage)")
        print(f"Domino size: {scene.domino_pixel_width:.1f} x {scene.domino_pixel_height:.1f} pixels")
        print(f"Total dominoes: {scene.num_dominoes}")
        print(f"Answer: {scene.answer} {scene.target_color} dominoes")
    else:
        print("Error: Expected output file not found")

    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
        print("Cleanup complete")
