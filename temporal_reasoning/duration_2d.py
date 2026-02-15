from manim import *
import random
import math
import os
import shutil
from pathlib import Path

# ============================================================================
# Setup directories for output files
# ============================================================================
Path("questions").mkdir(exist_ok=True)  # Video files
Path("solutions").mkdir(exist_ok=True)  # Answer text files
Path("question_text").mkdir(exist_ok=True)  # Question text files
Path("reasoning_traces").mkdir(exist_ok=True)  # Step-by-step reasoning

# ============================================================================
# Manim configuration
# ============================================================================
config.media_dir = "manim_output"
config.verbosity = "WARNING"
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.preview = False

# ============================================================================
# Usage: NUM_SHAPES [2-8] range for difficulty adjustment
# Example: NUM_SHAPES=2 python duration_2d.py
# ============================================================================


class duration_2d(ThreeDScene):
    """
    A 3D scene that generates a shape duration tracking puzzle:
    - Shows multiple 2D shapes appearing one at a time
    - Each shape is drawn for a specific duration
    - User must track and list the duration of each shape in order
    - Generates question video, solution, and reasoning trace
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ====================================================================
        # Set random seed for reproducibility of shape selection and timing
        # ====================================================================
        self.seed = random.randint(1000, 9999)
        random.seed(self.seed)

        # ====================================================================
        # Parameters - difficulty controls number of shapes to track
        # ====================================================================
        self.num_shapes = int(os.getenv("NUM_SHAPES", 5))

        # ====================================================================
        # Initialize event tracking for reasoning trace
        # This stores timestamped events as they occur during the video
        # ====================================================================
        self.scene_events = []

    def log_event(self, description):
        """
        Log a scene event with video timestamp.
        Uses Manim's internal renderer.time which tracks actual video playback time,
        not wall-clock execution time.

        Args:
            description: Human-readable description of what's happening
        """
        # Get current video time from Manim's renderer
        # self.renderer.time tracks the cumulative duration of all animations/waits
        current_time = self.renderer.time

        self.scene_events.append({"time": current_time, "description": description})

    def format_time(self, seconds):
        """
        Format seconds as M:SS for display in reasoning trace.

        Args:
            seconds: Time in seconds (float)

        Returns:
            String formatted as "M:SS" (e.g., "2:37")
        """
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def construct(self):
        """
        Main scene construction method.
        This is called by Manim to build and render the entire scene.
        """
        # ====================================================================
        # Camera setup - angled view for depth perception
        # ====================================================================
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)

        count = self.num_shapes

        # ====================================================================
        # Define shape library with human-readable names
        # Each shape is paired with its name for question generation
        # ====================================================================
        all_shapes_with_names = [
            (Circle(radius=1), "circle"),
            (Square(side_length=2), "square"),
            (Triangle().scale(1.5), "triangle"),
            (RegularPolygon(5).scale(1.2), "pentagon"),
            (Square(side_length=2).rotate(PI / 4), "diamond"),
            (RegularPolygon(6).scale(1.1), "hexagon"),
            (RegularPolygon(8).scale(1.1), "octagon"),
            (Star(5).scale(1.2), "star"),
        ]

        # ====================================================================
        # Color palette for shapes
        # Using distinctive colors to make shapes easily identifiable
        # ====================================================================
        all_colors = [WHITE, DARK_BROWN, RED, GREEN, PINK, BLUE, YELLOW, PURPLE, ORANGE]

        # ====================================================================
        # Predefined position pool for shape placement
        # Ensures shapes don't overlap and are well-distributed on screen
        # ====================================================================
        positions_pool = [
            LEFT * 3 + DOWN,
            RIGHT * 4 + DOWN,
            LEFT * 2 + UP,
            RIGHT * 2 + UP,
            ORIGIN,
            LEFT * 4 + UP,
            RIGHT * 4 + UP,
            RIGHT * 2 + DOWN,
            LEFT * 2 + DOWN,
            DOWN * 3,
            UP * 3,
            LEFT * 4,
        ]

        # ====================================================================
        # Randomly select shapes, colors, and positions
        # Using random.sample ensures no duplicates
        # ====================================================================
        chosen_shape_indices = []
        last_shape = None

        for _ in range(count):
            # Filter out the shape used in the previous step
            choices = [i for i in range(len(all_shapes_with_names)) if i != last_shape]

            current_shape = random.choice(choices)
            chosen_shape_indices.append(current_shape)

            # Update the tracker for the next iteration
            last_shape = current_shape

        chosen_color_indices = random.choices(range(len(all_colors)), k=count)
        chosen_positions = random.sample(positions_pool, count)

        # ====================================================================
        # Generate random durations for each shape (0.5s to 3.0s)
        # Varied durations make the puzzle challenging to track
        # ====================================================================
        durations = [round(random.uniform(0.5, 3.0), 1) for _ in range(count)]

        # ====================================================================
        # Track shapes and their durations for answer generation
        # ====================================================================
        shape_duration_pairs = []

        # ====================================================================
        # Log initial scene state
        # ====================================================================
        self.log_event("Video begins with empty scene, camera positioned at an angle")

        # ====================================================================
        # Animate shapes one by one
        # Each shape appears for its designated duration, then disappears
        # ====================================================================
        for idx in range(count):
            shape_original, shape_name = all_shapes_with_names[
                chosen_shape_indices[idx]
            ]
            shape = shape_original.copy()  # Copy to avoid modifying the template
            color = all_colors[chosen_color_indices[idx]]
            duration = durations[idx]
            position = chosen_positions[idx]

            # ================================================================
            # Configure shape appearance
            # Opacity makes shapes visually pleasing without being too bright
            # ================================================================
            shape.set_fill(color, opacity=0.6)
            shape.set_stroke(color)
            shape.move_to(position)

            # Store for answer generation
            shape_duration_pairs.append((shape_name, duration))

            # ================================================================
            # Log event BEFORE animation starts
            # ================================================================
            self.log_event(f"Shape {idx+1} ({shape_name}) begins drawing")

            # ================================================================
            # Animate the shape creation
            # The run_time parameter controls how long the drawing takes
            # ================================================================
            self.play(Create(shape), run_time=duration)

            # ================================================================
            # Log event AFTER animation completes
            # ================================================================
            self.log_event(
                f"Shape {idx+1} ({shape_name}) finishes drawing (duration: {duration:.1f}s)"
            )

        # ====================================================================
        # Transition to question
        # Clear all shapes before showing the question
        # ====================================================================
        self.log_event("All shapes complete, scene begins clearing")
        self.clear()
        self.wait(0.5)
        self.log_event("Scene cleared, ready to display question")

        # ====================================================================
        # Create and display question text
        # Fixed in frame so it stays visible during camera movements
        # ====================================================================
        question_lines = [
            "List the duration of each of the shapes from the beginning",
            "(from the start of drawing).",
            "",
            "Answer to 1 decimal point and list them with comma separated values:",
            "e.g., 3.2s, 1.5s, 1.0s",
        ]

        # ====================================================================
        # Create question text objects with proper formatting
        # First line is bold to emphasize the main question
        # ====================================================================
        question_texts = []
        line_height = 0.6
        start_y = 2.5

        for i, line in enumerate(question_lines):
            if line:  # Skip empty lines for spacing
                text = Text(line, font_size=24, weight=BOLD if i == 0 else NORMAL)
                text.move_to(UP * (start_y - i * line_height))
                question_texts.append(text)

        # Add all question text as fixed elements (won't move with camera)
        self.add_fixed_in_frame_mobjects(*question_texts)

        # ====================================================================
        # Animate question appearance
        # ====================================================================
        self.log_event("Question begins appearing on screen")
        self.play(*[FadeIn(text) for text in question_texts], run_time=1.0)
        self.wait(1.0)
        self.log_event("Question fully displayed")

        # ====================================================================
        # Show instruction reminder
        # Yellow color draws attention to important instruction
        # ====================================================================
        instruction_text = Text(
            "List the durations in the order the shapes appeared.",
            font_size=22,
            color=YELLOW,
        ).move_to(DOWN * 2.8)

        self.add_fixed_in_frame_mobjects(instruction_text)
        self.log_event("Instruction reminder appears")
        self.play(FadeIn(instruction_text, shift=UP * 0.3), run_time=0.8)
        self.wait(3)
        self.log_event("Question and instruction remain on screen for review")

        # ====================================================================
        # Generate answer: durations in order of appearance
        # ====================================================================
        duration_strings = [f"{duration}s" for _, duration in shape_duration_pairs]
        answer_string = ", ".join(duration_strings)

        # ====================================================================
        # Generate comprehensive reasoning trace
        # This explains the step-by-step solution process
        # ====================================================================
        self.build_reasoning_trace(shape_duration_pairs, answer_string)

        # ====================================================================
        # Save output files
        # ====================================================================
        # Solution file (just the answer)
        with open(
            f"solutions/duration2d_n{self.num_shapes}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(answer_string)

        # Question text file
        question_text_content = (
            "List the duration of each of the shapes from the beginning (from the start of drawing).\n"
            "Answer to 1 decimal point and list them with comma separated values: e.g., 3.2s, 1.5s, 1.0s\n"
            "List the durations in the order the shapes appeared."
        )
        with open(
            f"question_text/duration2d_n{self.num_shapes}_seed{self.seed}.txt", "w"
        ) as f:
            f.write(question_text_content)

        # Reasoning trace file
        with open(
            f"reasoning_traces/duration2d_n{self.num_shapes}_seed{self.seed}.txt", "w"
        ) as f:
            f.write("\n".join(self.reasoning_trace))

    def build_reasoning_trace(self, shape_duration_pairs, answer_string):
        """
        Build a comprehensive, step-by-step reasoning trace.
        This explains how to solve the puzzle systematically.

        Args:
            shape_duration_pairs: List of (shape_name, duration) tuples in order
            answer_string: Final formatted answer string
        """
        self.reasoning_trace = []

        # ====================================================================
        # Introduction with question statement
        # ====================================================================
        self.reasoning_trace.append(
            "**Question:** List the duration of each of the shapes from the beginning (from the start of drawing). Answer to 1 decimal point and list them with comma separated values."
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Let's solve this step by step.")
        self.reasoning_trace.append("")

        # ====================================================================
        # Scene description with timestamps
        # This helps the reader understand what happened and when
        # ====================================================================
        self.reasoning_trace.append("### Scene Description")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            "The video shows a sequence of 2D shapes appearing one at a time. Each shape is drawn over a specific duration, then the scene clears before the next shape appears."
        )
        self.reasoning_trace.append("")

        for event in self.scene_events:
            time_str = self.format_time(event["time"])
            self.reasoning_trace.append(f"At {time_str}, {event['description']}")

        self.reasoning_trace.append("")

        # ====================================================================
        # Step 1: Understand the task
        # ====================================================================
        self.reasoning_trace.append("### Step 1: Understand what we need to track")
        self.reasoning_trace.append(
            f"This video tests our ability to track the duration of multiple shapes appearing sequentially."
        )
        self.reasoning_trace.append(f"")
        self.reasoning_trace.append(f"Key observations:")
        self.reasoning_trace.append(
            f"- There are **{len(shape_duration_pairs)} shapes** in total"
        )
        self.reasoning_trace.append(
            f"- Each shape appears one at a time (never simultaneously)"
        )
        self.reasoning_trace.append(
            f"- Each shape is drawn over a specific time period"
        )
        self.reasoning_trace.append(
            f"- We need to track how long each drawing animation takes"
        )
        self.reasoning_trace.append("")

        # ====================================================================
        # Step 2: Track each shape's duration
        # ====================================================================
        self.reasoning_trace.append(
            "### Step 2: Observe and record each shape's duration"
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            "Let's go through each shape in the order they appeared:"
        )
        self.reasoning_trace.append("")

        cumulative_time = 0.0
        for i, (shape_name, duration) in enumerate(shape_duration_pairs):
            start_time = cumulative_time
            end_time = cumulative_time + duration

            self.reasoning_trace.append(f"**Shape {i+1}: {shape_name.capitalize()}**")
            self.reasoning_trace.append(f"  - Drawing starts at: {start_time:.1f}s")
            self.reasoning_trace.append(f"  - Drawing ends at: {end_time:.1f}s")
            self.reasoning_trace.append(f"  - **Duration: {duration:.1f} seconds**")
            self.reasoning_trace.append("")

            cumulative_time = end_time

        # ====================================================================
        # Step 3: Format the answer
        # ====================================================================
        self.reasoning_trace.append(
            "### Step 3: Format the answer according to requirements"
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append("The question asks for:")
        self.reasoning_trace.append("- Durations to 1 decimal point")
        self.reasoning_trace.append("- Comma-separated values")
        self.reasoning_trace.append("- Format: X.Xs, Y.Ys, Z.Zs")
        self.reasoning_trace.append("- Listed in the order the shapes appeared")
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Compiling the answer:")

        for i, (shape_name, duration) in enumerate(shape_duration_pairs):
            if i < len(shape_duration_pairs) - 1:
                self.reasoning_trace.append(
                    f"  - Shape {i+1} ({shape_name}): {duration:.1f}s, (add comma)"
                )
            else:
                self.reasoning_trace.append(
                    f"  - Shape {i+1} ({shape_name}): {duration:.1f}s (last entry, no comma)"
                )

        self.reasoning_trace.append("")

        # ====================================================================
        # Final answer
        # ====================================================================
        self.reasoning_trace.append("### Final Answer")
        self.reasoning_trace.append("")
        self.reasoning_trace.append(
            f"Listing all durations in order: **{answer_string}**"
        )
        self.reasoning_trace.append("")
        self.reasoning_trace.append(f"\\boxed{{{answer_string}}}")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    # ========================================================================
    # Generate the duration 2D video
    # ========================================================================
    scene = duration_2d()
    scene.render()

    # ========================================================================
    # Move output file to questions directory with descriptive name
    # Filename includes number of shapes and seed for reproducibility
    # ========================================================================
    output = Path("manim_output/videos/1080p30/duration_2d.mp4")
    if output.exists():
        filename = f"duration2d_n{scene.num_shapes}_seed{scene.seed}.mp4"
        shutil.move(str(output), f"questions/{filename}")
        print(f"✓ Video saved: questions/{filename}")
    else:
        # ====================================================================
        # Debug output if video file not found in expected location
        # Helps diagnose Manim output path issues
        # ====================================================================
        videos_dir = Path("manim_output/videos")
        if videos_dir.exists():
            print(f"Available folders in videos/: {list(videos_dir.iterdir())}")
            for folder in videos_dir.iterdir():
                if folder.is_dir():
                    subfolder = folder / "1080p30"
                    if subfolder.exists():
                        print(f"Files in {subfolder}: {list(subfolder.iterdir())}")
        else:
            print("manim_output/videos directory doesn't exist")

    # ========================================================================
    # Final cleanup - remove temporary Manim output directory
    # This keeps the working directory clean by removing intermediate files
    # ========================================================================
    if os.path.exists("manim_output"):
        shutil.rmtree("manim_output")
